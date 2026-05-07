"""
geo_tvt/model/predictor.py
Two-stage TVT prediction system:

Stage 1 — CatBoost Baseline
  Fast, strong baseline using engineered sequence features + geological priors.
  Good for competition Phase 1.

Stage 2 — Geological-Aware Transformer
  Sequence model that fuses typewell context embedding with drilling telemetry.
  Handles the NaN TVT prediction problem as autoregressive continuation.
  Good for competition Phase 2 / final submission.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import sys

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SEQUENCE_WINDOW, CATBOOST_PARAMS, MODEL_DIR


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_sequence_features(df: pd.DataFrame,
                                geo_prior: Optional[dict] = None) -> pd.DataFrame:
    """
    Build sequence features for the CatBoost baseline.
    
    Inputs: drilling log DataFrame with columns:
      MD, X, Y, Z, GR, TVT_input (may have NaN in prediction zone)
    
    Adds lag features, rolling stats, rate-of-change, geological priors.
    """
    feat = df.copy()
    base_cols = [c for c in ["GR", "MD", "X", "Y", "Z"] if c in feat.columns]

    for col in base_cols:
        # Lags
        for lag in [1, 2, 3, 5, 10]:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)
        # Rolling stats
        for w in [5, 10, 20]:
            feat[f"{col}_roll_mean{w}"] = feat[col].rolling(w, min_periods=1).mean()
            feat[f"{col}_roll_std{w}"]  = feat[col].rolling(w, min_periods=1).std().fillna(0)
        # Rate of change
        feat[f"{col}_roc"] = feat[col].diff().fillna(0)

    # TVT history features (only from known region)
    if "TVT_input" in feat.columns:
        tvt_known = feat["TVT_input"]
        feat["tvt_lag1"]  = tvt_known.shift(1)
        feat["tvt_lag2"]  = tvt_known.shift(2)
        feat["tvt_lag5"]  = tvt_known.shift(5)
        feat["tvt_delta"] = tvt_known.diff().fillna(0)
        feat["tvt_trend"] = tvt_known.rolling(10, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )

    # Trajectory features
    if all(c in feat.columns for c in ["X", "Y", "Z"]):
        feat["horiz_dist"] = np.sqrt(feat["X"]**2 + feat["Y"]**2)
        feat["inclination_proxy"] = feat["Z"].diff().abs().rolling(5, min_periods=1).mean()

    # Position within well
    if "well_id" in feat.columns and "MD" in feat.columns:
        md_min = feat.groupby("well_id")["MD"].transform("min")
        md_max = feat.groupby("well_id")["MD"].transform("max")
        span = (md_max - md_min).replace(0, 1)
        feat["md_frac"] = (feat["MD"] - md_min) / span

    # TVT forward/backfill features
    if "TVT_input" in feat.columns:
        feat["tvt_ffill"] = feat.groupby("well_id")["TVT_input"].ffill().fillna(0)
        feat["tvt_bfill"] = feat.groupby("well_id")["TVT_input"].bfill().fillna(0)
        feat["tvt_known_flag"] = feat["TVT_input"].notna().astype(int)
        feat["tvt_delta2"] = feat["tvt_lag1"] - feat["tvt_lag2"]
        feat["tvt_delta5"] = feat["tvt_lag1"] - feat["tvt_lag5"]

    # More GR derived features
    if "GR" in feat.columns:
        feat["gr_delta1"] = feat["GR"] - feat["GR_lag1"]
        feat["gr_delta2"] = feat["GR_lag1"] - feat["GR_lag2"]
        feat["gr_delta5"] = feat["GR_lag1"] - feat["GR_lag5"]
        feat["gr_roll_range10"] = feat["gr_roll_mean10"] - feat["gr_roll_mean5"]

    # Geological prior features (constant per well, but very powerful)
    if geo_prior:
        for k, v in geo_prior.items():
            if isinstance(v, (int, float)):
                feat[f"prior_{k}"] = float(v)

    # Add well-level metadata and cluster features
    feat = _add_well_summary_features(feat)
    feat = _add_well_cluster_features(feat)

    feat = feat.fillna(feat.median(numeric_only=True))
    return feat


def _add_well_summary_features(feat: pd.DataFrame) -> pd.DataFrame:
    if "well_id" not in feat.columns:
        return feat

    agg_defs = {
        "MD": ["min", "max"],
        "GR": ["mean", "std", "min", "max"],
        "Z": ["min", "max"],
        "TVT_input": ["mean", "std"],
    }
    summary = feat.groupby("well_id").agg(agg_defs)
    summary.columns = [f"well_{col}_{fn}" for col, fn in summary.columns]
    summary = summary.fillna(0)

    summary["well_md_span"] = (summary["well_MD_max"] - summary["well_MD_min"]).replace(0, 1)
    summary["well_gr_range"] = (summary["well_GR_max"] - summary["well_GR_min"]).fillna(0)
    summary["well_z_range"] = (summary["well_Z_max"] - summary["well_Z_min"]).fillna(0)
    summary["well_tvt_known_frac"] = feat.groupby("well_id")["TVT_input"].apply(
        lambda s: float(s.notna().mean()) if len(s) > 0 else 0.0
    )
    summary["well_row_count"] = feat.groupby("well_id")["MD"].size().astype(float)

    summary = summary.reset_index()
    feat = feat.merge(summary, on="well_id", how="left")
    return feat


def _add_well_cluster_features(feat: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    if "well_id" not in feat.columns:
        return feat

    cluster_features = [
        "well_GR_mean", "well_GR_std", "well_gr_range",
        "well_tvt_known_frac", "well_z_range", "well_md_span",
    ]
    if not all(col in feat.columns for col in cluster_features):
        return feat

    well_summary = feat["well_id"].drop_duplicates().to_frame()
    well_summary = well_summary.merge(
        feat.groupby("well_id")[cluster_features].first().reset_index(),
        on="well_id", how="left"
    ).fillna(0)

    if len(well_summary) < n_clusters:
        feat["well_cluster_id"] = 0
        return feat

    try:
        from sklearn.cluster import KMeans
        numeric = well_summary[cluster_features].astype(float)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        well_summary["well_cluster_id"] = kmeans.fit_predict(numeric)
    except Exception:
        feat["well_cluster_id"] = 0
        return feat

    feat = feat.merge(well_summary[["well_id", "well_cluster_id"]], on="well_id", how="left")
    feat["well_cluster_id"] = feat["well_cluster_id"].fillna(0).astype(int)
    return feat


# ─── CatBoost Baseline ───────────────────────────────────────────────────────

def train_catboost(X_train: pd.DataFrame, y_train: np.ndarray,
                   X_val: pd.DataFrame, y_val: np.ndarray):
    """Train CatBoost TVT regressor."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("pip install catboost")

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        use_best_model=True,
    )
    return model


def predict_catboost(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def save_catboost(model, name: str = "catboost_tvt"):
    path = MODEL_DIR / f"{name}.cbm"
    model.save_model(str(path))
    print(f"[model] Saved CatBoost → {path}")


def load_catboost(name: str = "catboost_tvt"):
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("pip install catboost")
    path = MODEL_DIR / f"{name}.cbm"
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


# ─── Transformer TVT Predictor ───────────────────────────────────────────────

if nn is None:
    class GeoTVTTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GeoTVTTransformer")

        def forward(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GeoTVTTransformer")
else:
    class GeoTVTTransformer(nn.Module):
        """
        Geological-aware sequence model for TVT prediction.
        
        Fuses:
          1. Drilling telemetry sequence (GR, MD, XYZ, past TVT)
          2. Typewell context embedding (from TypewellEncoder)
          3. Geological prior vector (from prior engine)
        
        Uses a causal transformer to autoregressively predict TVT values
        in the NaN zone (the actual competition prediction target).
        """

    def __init__(
        self,
        n_telemetry_features: int = 8,
        typewell_embed_dim: int = 64,
        n_prior_features: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        seq_len: int = SEQUENCE_WINDOW,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len

        # Project telemetry to model dim
        self.telemetry_proj = nn.Linear(n_telemetry_features, d_model)

        # Fuse typewell embedding
        self.typewell_proj  = nn.Linear(typewell_embed_dim, d_model)

        # Fuse geological priors
        self.prior_proj     = nn.Linear(n_prior_features, d_model)

        # Positional encoding
        self.pos_enc = nn.Embedding(seq_len + 10, d_model)

        # Transformer decoder (causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),  # predict next TVT
        )

        # Uncertainty head (aleatoric)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # σ > 0
        )

    def forward(
        self,
        telemetry: torch.Tensor,       # [B, T, n_telemetry_features]
        typewell_emb: torch.Tensor,    # [B, typewell_embed_dim]
        prior_vec: torch.Tensor,       # [B, n_prior_features]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          tvt_pred:    [B, T, 1] predicted TVT at each step
          uncertainty: [B, T, 1] predicted uncertainty (std dev)
        """
        B, T, _ = telemetry.shape

        # Project inputs
        x = self.telemetry_proj(telemetry)

        # Add typewell context (broadcast across time)
        tw = self.typewell_proj(typewell_emb).unsqueeze(1).expand(-1, T, -1)
        x = x + tw

        # Add geological prior (broadcast across time)
        pr = self.prior_proj(prior_vec).unsqueeze(1).expand(-1, T, -1)
        x = x + pr

        # Positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_enc(positions)

        # Causal mask (each step can only see previous steps)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        # Transformer
        out = self.transformer(x, mask=mask)

        return self.output_head(out), self.uncertainty_head(out)


def autoregressive_predict(
    model: GeoTVTTransformer,
    known_telemetry: np.ndarray,     # [T_known, features]
    typewell_emb: np.ndarray,        # [embed_dim]
    prior_vec: np.ndarray,           # [prior_features]
    n_steps: int = 10,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Autoregressively predict n_steps of TVT beyond the known region.
    This directly handles the NaN zone in the competition data.
    
    Returns: (predictions [n_steps], uncertainties [n_steps])
    """
    if torch is None:
        raise ImportError("PyTorch is required for autoregressive_predict")
    model.eval()

    telemetry = torch.tensor(known_telemetry, dtype=torch.float32, device=device)
    tw_emb    = torch.tensor(typewell_emb, dtype=torch.float32, device=device).unsqueeze(0)
    pr_vec    = torch.tensor(prior_vec, dtype=torch.float32, device=device).unsqueeze(0)

    preds, stds = [], []

    with torch.no_grad():
        current_seq = telemetry[-SEQUENCE_WINDOW:].unsqueeze(0)  # [1, W, F]

        for _ in range(n_steps):
            pred, unc = model(current_seq, tw_emb, pr_vec)
            next_tvt = pred[0, -1, 0].item()
            next_std = unc[0, -1, 0].item()
            preds.append(next_tvt)
            stds.append(next_std)

            # Roll window forward — update TVT feature in last position
            new_step = current_seq[0, -1, :].clone()
            # Assume TVT is feature index 0 in telemetry
            new_step[0] = next_tvt
            current_seq = torch.cat([
                current_seq[:, 1:, :],
                new_step.unsqueeze(0).unsqueeze(0)
            ], dim=1)

    return np.array(preds), np.array(stds)
