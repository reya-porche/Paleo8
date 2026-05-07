"""
geo_tvt/physics/hybrid_predictor.py
Physics + ML hybrid prediction system.

Architecture:
  Step 1 — Physics baseline:
    Predict next TVT using known physical laws:
      - Structural continuity (TVT follows the formation dip)
      - Trajectory momentum (well direction doesn't change instantly)
      - GR-guided transition detection
    This gives a physically plausible prediction.

  Step 2 — Neural residual:
    A small neural net or boosted model predicts the correction term:
      TVT_final = TVT_physics + learned_residual

Why this is powerful:
  - Physics constrains the prediction to physically plausible space
  - ML learns the deviations from perfect physics (real geology is noisy)
  - The residual is much easier to predict than absolute TVT
  - Uncertainty estimation becomes natural (residual variance)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR


# ─── Physics Baseline ────────────────────────────────────────────────────────

class PhysicsBaseline:
    """
    Physics-based TVT continuation.

    Predicts TVT using:
      1. Structural dip extrapolation (TVT trend from recent history)
      2. Trajectory-adjusted continuation (Z movement implies TVT movement)
      3. GR-transition detection (large GR changes signal formation boundaries)

    This is not ML. It encodes actual drilling physics.
    """

    def __init__(
        self,
        dip_window:     int   = 20,    # rows to estimate structural dip
        gr_sensitivity: float = 0.05,  # TVT adjustment per GR unit change
        max_delta:      float = 3.0,   # maximum TVT change per step
        smoothing:      float = 0.3,   # exponential smoothing weight
    ):
        self.dip_window     = dip_window
        self.gr_sensitivity = gr_sensitivity
        self.max_delta      = max_delta
        self.smoothing      = smoothing

    def predict_continuation(
        self,
        known_tvt:  np.ndarray,   # known TVT values
        known_gr:   np.ndarray,   # GR in known zone
        known_z:    np.ndarray,   # Z in known zone
        pred_gr:    np.ndarray,   # GR in prediction zone
        pred_z:     np.ndarray,   # Z in prediction zone
        pred_md:    Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict TVT continuation using physics.

        Returns: (predictions, confidence_scores)
        """
        n_pred = len(pred_gr)
        preds  = np.zeros(n_pred)
        conf   = np.zeros(n_pred)

        if len(known_tvt) < 2:
            anchor = known_tvt[-1] if len(known_tvt) > 0 else 0.0
            return np.full(n_pred, anchor), np.full(n_pred, 0.1)

        # ── Estimate structural dip ──────────────────────────────────────────
        window = min(self.dip_window, len(known_tvt))
        recent_tvt = known_tvt[-window:]
        recent_z   = known_z[-window:]

        # Fit TVT vs Z relationship (formation dip in the local coordinate frame)
        if np.std(recent_z) > 0.01:
            dip_coeff = float(np.polyfit(recent_z, recent_tvt, 1)[0])
        else:
            dip_coeff = 0.0

        # Fit TVT trend vs MD steps
        x_steps = np.arange(window, dtype=float)
        tvt_slope = float(np.polyfit(x_steps, recent_tvt, 1)[0])

        # ── Estimate GR→TVT sensitivity ──────────────────────────────────────
        # In the known zone, how much did TVT change per GR unit change?
        if len(known_gr) >= window:
            recent_gr = known_gr[-window:]
            gr_changes   = np.diff(recent_gr)
            tvt_changes  = np.diff(recent_tvt)
            if np.std(gr_changes) > 0.1:
                gr_tvt_corr = float(np.polyfit(gr_changes, tvt_changes, 1)[0])
            else:
                gr_tvt_corr = 0.0
        else:
            gr_tvt_corr = 0.0

        # ── Predict step by step ─────────────────────────────────────────────
        tvt_buffer = list(known_tvt[-window:])
        z_buffer   = list(known_z[-window:])
        gr_buffer  = list(known_gr[-window:])

        prev_delta = tvt_slope  # start with known trend

        for step in range(n_pred):
            # Physics components
            z_delta  = pred_z[step] - z_buffer[-1] if step < len(pred_z) else 0.0
            gr_delta = pred_gr[step] - gr_buffer[-1]

            # Component 1: Structural continuation (dip × Z movement)
            dip_component = dip_coeff * z_delta

            # Component 2: Trend continuation (momentum)
            trend_component = tvt_slope

            # Component 3: GR-driven adjustment
            gr_component = gr_tvt_corr * gr_delta

            # Weighted combination
            raw_delta = (
                0.5 * trend_component +
                0.3 * dip_component   +
                0.2 * gr_component
            )

            # Exponential smoothing with previous delta
            smoothed_delta = (
                self.smoothing * raw_delta +
                (1 - self.smoothing) * prev_delta
            )

            # Physical plausibility constraint
            clipped_delta = np.clip(smoothed_delta, -self.max_delta, self.max_delta)

            next_tvt = tvt_buffer[-1] + clipped_delta

            preds[step] = next_tvt
            prev_delta  = clipped_delta

            # Update buffers
            tvt_buffer.append(next_tvt)
            z_buffer.append(pred_z[step] if step < len(pred_z) else z_buffer[-1])
            gr_buffer.append(pred_gr[step])

            # Confidence decreases with steps
            conf[step] = float(np.exp(-step * 0.05))

        return preds, conf


# ─── Residual Learner ─────────────────────────────────────────────────────────

class ResidualLearner:
    """
    Learns the correction term: TVT_true - TVT_physics.

    The residual is much easier to predict than absolute TVT because:
      - Physics already handles the bulk of the signal
      - Residuals are smaller, lower-variance, zero-mean (mostly)
      - ML needs to learn only the exceptions: faults, transitions, noise

    Features for residual learning:
      - Physics prediction itself (how confident the physics is)
      - GR features (what lithology is causing deviation)
      - Trajectory features (is the well bending?)
      - Recent residual history (systematic bias?)
    """

    def __init__(self):
        self.model  = None
        self.scaler = None
        self._fitted = False

    def build_residual_features(
        self,
        df:          pd.DataFrame,
        physics_col: str   = "tvt_physics_pred",
        tvt_col:     str   = "TVT",
        gr_col:      str   = "GR",
        z_col:       str   = "Z",
    ) -> pd.DataFrame:
        """
        Build features for residual prediction.
        Only works in the known TVT zone (where true TVT exists).
        """
        feat = df.copy()

        if tvt_col in feat.columns and physics_col in feat.columns:
            feat["residual_target"] = feat[tvt_col] - feat[physics_col]

        # Residual features: things the physics missed
        if gr_col in feat.columns:
            gr = feat[gr_col]
            feat["res_gr_spike"]  = (gr - gr.rolling(10, min_periods=1).mean()).abs()
            feat["res_gr_accel"]  = gr.diff().diff().fillna(0)

        if z_col in feat.columns:
            z = feat[z_col]
            feat["res_z_accel"] = z.diff().diff().fillna(0)

        if physics_col in feat.columns:
            feat["res_physics_uncertainty"] = feat[physics_col].diff().abs().fillna(0)

        return feat

    def fit(
        self,
        train_df:    pd.DataFrame,
        physics_col: str = "tvt_physics_pred",
        tvt_col:     str = "TVT",
    ) -> "ResidualLearner":
        """Train the residual corrector."""
        try:
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(
                iterations=300, learning_rate=0.05, depth=5,
                verbose=0, random_seed=42,
                loss_function="MAE",  # more robust to outliers
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, loss="absolute_error", random_state=42
            )

        feat_df = self.build_residual_features(train_df, physics_col, tvt_col)
        res_cols = [c for c in feat_df.columns if c.startswith("res_")]
        if not res_cols:
            print("[hybrid] No residual feature columns found, skipping")
            return self

        mask = feat_df["residual_target"].notna() & feat_df[res_cols[0]].notna()
        if mask.sum() < 20:
            print("[hybrid] Not enough samples for residual learning")
            return self

        y_check = feat_df[mask]["residual_target"].values
        if y_check.std() < 1e-6:
            print("[hybrid] Residuals have near-zero variance (physics=actual in known zone), skipping residual learner")
            return self

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X = feat_df[mask][res_cols].select_dtypes(include=[np.number]).fillna(0)
        y = feat_df[mask]["residual_target"].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        print(f"[hybrid] Residual learner trained on {mask.sum()} samples")
        return self

    def predict_residual(
        self, feat_df: pd.DataFrame, physics_col: str = "tvt_physics_pred"
    ) -> np.ndarray:
        """Predict residual correction."""
        if not self._fitted:
            return np.zeros(len(feat_df))

        aug = self.build_residual_features(feat_df, physics_col)
        res_cols = [c for c in aug.columns if c.startswith("res_")]
        X = aug[res_cols].select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ─── Full Hybrid Predictor ────────────────────────────────────────────────────

class HybridTVTPredictor:
    """
    Complete physics + ML hybrid.

    Usage:
      predictor = HybridTVTPredictor()
      predictor.fit(train_df)
      predictions = predictor.predict(test_df)
    """

    def __init__(self, **physics_kwargs):
        self.physics  = PhysicsBaseline(**physics_kwargs)
        self.residual = ResidualLearner()

    def fit(
        self,
        train_df: pd.DataFrame,
        well_col: str = "well_id",
        tvt_col:  str = "TVT",
        gr_col:   str = "GR",
        z_col:    str = "Z",
        md_col:   str = "MD",
    ) -> "HybridTVTPredictor":
        """Fit physics baseline + residual learner."""
        print("[hybrid] Generating physics predictions on training data...")
        train_with_physics = self._add_physics_predictions(
            train_df, well_col, tvt_col, gr_col, z_col
        )
        print("[hybrid] Training residual learner...")
        self.residual.fit(train_with_physics, tvt_col=tvt_col)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        well_col: str = "well_id",
        tvt_col:  str = "TVT_input",
        gr_col:   str = "GR",
        z_col:    str = "Z",
    ) -> pd.DataFrame:
        """Predict TVT = physics + residual."""
        out = self._add_physics_predictions(df, well_col, tvt_col, gr_col, z_col)
        residuals = self.residual.predict_residual(out, physics_col="tvt_physics_pred")
        out["TVT_pred_hybrid"]    = out["tvt_physics_pred"] + residuals
        out["TVT_pred_physics"]   = out["tvt_physics_pred"]
        out["TVT_pred_residual"]  = residuals
        return out

    def _add_physics_predictions(
        self,
        df: pd.DataFrame,
        well_col: str,
        tvt_col:  str,
        gr_col:   str,
        z_col:    str,
    ) -> pd.DataFrame:
        """Generate per-well physics baseline predictions."""
        parts = []
        for _, grp in df.groupby(well_col, sort=False):
            grp = grp.copy()
            tvt = grp[tvt_col].values if tvt_col in grp.columns else np.zeros(len(grp))
            gr  = grp[gr_col].fillna(grp[gr_col].median()).values if gr_col in grp.columns else np.zeros(len(grp))
            z   = grp[z_col].values if z_col in grp.columns else np.zeros(len(grp))

            known_mask = np.isfinite(tvt)
            phys_preds = tvt.copy()

            if known_mask.sum() >= 2 and (~known_mask).sum() > 0:
                known_tvt = tvt[known_mask]
                known_gr  = gr[known_mask]
                known_z   = z[known_mask]
                pred_gr   = gr[~known_mask]
                pred_z    = z[~known_mask]
                n_pred    = (~known_mask).sum()

                preds, _ = self.physics.predict_continuation(
                    known_tvt, known_gr, known_z, pred_gr, pred_z
                )
                phys_preds[~known_mask] = preds

            grp["tvt_physics_pred"] = phys_preds
            parts.append(grp)

        return pd.concat(parts, ignore_index=True)
