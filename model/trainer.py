"""
geo_tvt/model/trainer.py
Training pipeline for both CatBoost baseline and Transformer model.
Includes cross-well validation split logic appropriate for this competition.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, SEQUENCE_WINDOW
from model.predictor import (
    engineer_sequence_features,
    train_catboost, predict_catboost, save_catboost,
)
from geo_tvt.clustering.well_clustering import (
    WellClusterer,
    build_well_summary_features,
    train_per_cluster_models,
)


# ─── Data Preparation ────────────────────────────────────────────────────────

def load_competition_data(data_path: str) -> pd.DataFrame:
    """
    Load competition drilling data CSV.
    Expected columns (adjust to actual competition format):
      well_id, MD, X, Y, Z, GR, TVT_input, [formation labels], [surface depths]
    """
    df = pd.read_csv(data_path)
    print(f"[trainer] Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"[trainer] Wells: {df['well_id'].nunique() if 'well_id' in df.columns else 'N/A'}")
    return df


def split_wells(df: pd.DataFrame, val_frac: float = 0.2,
                seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by well ID (not by row) to prevent data leakage.
    A model trained on one well should predict unseen wells.
    """
    if "well_id" not in df.columns:
        # Fallback: time-series split
        n = int(len(df) * (1 - val_frac))
        return df.iloc[:n], df.iloc[n:]

    rng = np.random.RandomState(seed)
    wells = df["well_id"].unique()
    rng.shuffle(wells)
    n_val = max(1, int(len(wells) * val_frac))
    val_wells  = set(wells[:n_val])
    train_wells = set(wells[n_val:])

    train_df = df[df["well_id"].isin(train_wells)].copy()
    val_df   = df[df["well_id"].isin(val_wells)].copy()
    print(f"[trainer] Train: {len(train_wells)} wells ({len(train_df):,} rows)")
    print(f"[trainer] Val:   {len(val_wells)} wells ({len(val_df):,} rows)")
    return train_df, val_df


def prepare_catboost_features(df: pd.DataFrame,
                               geo_priors: Optional[dict] = None,
                               target_col: str = "TVT") -> tuple:
    """
    Build feature matrix and target vector for CatBoost.
    Drops rows where target is NaN (those are the prediction zone).
    """
    feat_df = engineer_sequence_features(df, geo_prior=geo_priors)

    # Target
    if target_col not in feat_df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(feat_df.columns)}")

    mask = feat_df[target_col].notna()
    X = feat_df[mask].drop(columns=[target_col, "well_id"], errors="ignore")
    y = feat_df[mask][target_col].values

    # Remove non-numeric
    X = X.select_dtypes(include=[np.number]).fillna(0)

    return X, y


def _prepare_transformer_dataset(
    df: pd.DataFrame,
    geo_priors: Optional[dict] = None,
    seq_len: int = SEQUENCE_WINDOW,
    target_col: str = "TVT_input",
):
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for transformer training and dataset preparation.")

    feat_df = engineer_sequence_features(df, geo_prior=geo_priors).fillna(0)
    telemetry_cols = [
        c for c in ["GR", "MD", "X", "Y", "Z", "tvt_lag1", "tvt_lag2", "tvt_lag5", "tvt_delta"]
        if c in feat_df.columns
    ]
    if len(telemetry_cols) < 4:
        raise ValueError("Not enough telemetry columns available for transformer training.")
    if target_col not in feat_df.columns:
        raise ValueError(f"Target column '{target_col}' not found for transformer training.")

    X_windows = []
    y_windows = []

    for _, grp in feat_df.groupby("well_id"):
        arr_tel = grp[telemetry_cols].values.astype(np.float32)
        arr_tvt = grp[target_col].values.astype(np.float32)
        n = len(grp)
        for start in range(0, n - seq_len):
            window_tvt = arr_tvt[start:start + seq_len]
            if np.isnan(window_tvt).any():
                continue
            X_windows.append(arr_tel[start:start + seq_len])
            y_windows.append(window_tvt.reshape(seq_len, 1))

    if not X_windows:
        raise ValueError("No valid transformer training windows found. Check your TVT_input coverage and seq_len.")

    X = torch.tensor(np.stack(X_windows), dtype=torch.float32)
    y = torch.tensor(np.stack(y_windows), dtype=torch.float32)
    typewell_emb = torch.zeros((len(X_windows), 64), dtype=torch.float32)
    prior_vec = torch.zeros((len(X_windows), 16), dtype=torch.float32)

    return X, typewell_emb, prior_vec, y


def train_transformer_model(
    data_path: str,
    geo_priors: Optional[dict] = None,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-4,
    device: Optional[str] = None,
) -> dict:
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for transformer training.")

    from model.predictor import GeoTVTTransformer

    df = load_competition_data(data_path)
    train_df, val_df = split_wells(df)

    print("[trainer] Preparing transformer sequences...")
    X_train, tw_train, prior_train, y_train = _prepare_transformer_dataset(train_df, geo_priors)
    X_val, tw_val, prior_val, y_val = _prepare_transformer_dataset(val_df, geo_priors)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoTVTTransformer(
        n_telemetry_features=X_train.shape[2],
        seq_len=X_train.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.L1Loss()

    train_dataset = torch.utils.data.TensorDataset(X_train, tw_train, prior_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, tw_val, prior_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, btw, bprior, by in train_loader:
            bx = bx.to(device)
            btw = btw.to(device)
            bprior = bprior.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            preds, _ = model(bx, btw, bprior)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0
        with torch.no_grad():
            for bx, btw, bprior, by in val_loader:
                bx = bx.to(device)
                btw = btw.to(device)
                bprior = bprior.to(device)
                by = by.to(device)
                preds, _ = model(bx, btw, bprior)
                loss = criterion(preds, by)
                val_loss += loss.item() * bx.size(0)
                val_mae += torch.mean(torch.abs(preds - by)).item() * bx.size(0)
                val_rmse += torch.mean((preds - by) ** 2).item() * bx.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_rmse = float(np.sqrt(val_rmse / len(val_loader.dataset)))

        print(f"[trainer] Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_mae={val_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_state = model.state_dict()

    model_path = MODEL_DIR / "transformer_tvt.pt"
    torch.save(best_state, model_path)
    print(f"[trainer] Saved transformer model → {model_path}")

    return {
        "mae": float(best_val_mae),
        "rmse": float(best_val_rmse),
        "model_path": str(model_path),
    }


# ─── Training Orchestration ──────────────────────────────────────────────────

def train_baseline(data_path: str, geo_priors: Optional[dict] = None) -> dict:
    """
    Full training run for CatBoost baseline.
    Returns evaluation metrics.
    """
    df = load_competition_data(data_path)
    train_df, val_df = split_wells(df)

    print("[trainer] Engineering features...")
    X_train, y_train = prepare_catboost_features(train_df, geo_priors)
    X_val,   y_val   = prepare_catboost_features(val_df, geo_priors)

    # Align columns
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    print(f"[trainer] Feature matrix: {X_train.shape}")
    print("[trainer] Training CatBoost...")
    model = train_catboost(X_train, y_train, X_val, y_val)
    save_catboost(model)

    # Evaluate
    val_preds = predict_catboost(model, X_val)
    mae  = float(np.mean(np.abs(val_preds - y_val)))
    rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))

    print(f"[trainer] Val MAE:  {mae:.4f}")
    print(f"[trainer] Val RMSE: {rmse:.4f}")

    # Feature importance
    fi = pd.Series(
        model.get_feature_importance(),
        index=X_train.columns
    ).sort_values(ascending=False).head(20)
    print("\n[trainer] Top 20 features:")
    print(fi.to_string())

    return {"mae": mae, "rmse": rmse, "n_features": X_train.shape[1]}


def cross_validate_baseline(
    data_path: str,
    geo_priors: Optional[dict] = None,
    n_splits: int = 5,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Run repeated well-level validation splits for CatBoost baseline."""
    df = load_competition_data(data_path)
    maes, rmses = [], []

    for fold in range(n_splits):
        train_df, val_df = split_wells(df, val_frac=val_frac, seed=seed + fold)
        X_train, y_train = prepare_catboost_features(train_df, geo_priors)
        X_val, y_val = prepare_catboost_features(val_df, geo_priors)
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

        model = train_catboost(X_train, y_train, X_val, y_val)
        val_preds = predict_catboost(model, X_val)

        mae = float(np.mean(np.abs(val_preds - y_val)))
        rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
        maes.append(mae)
        rmses.append(rmse)

        print(f"[trainer] Fold {fold + 1}/{n_splits}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    return {
        "n_splits": n_splits,
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes, ddof=1)),
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses, ddof=1)),
        "maes": maes,
        "rmses": rmses,
    }


def train_clustered_baseline(
    data_path: str,
    geo_priors: Optional[dict] = None,
    n_clusters: int = 5,
) -> dict:
    """Train a cluster-aware CatBoost baseline by adding well cluster features."""
    df = load_competition_data(data_path)
    summary = build_well_summary_features(df)
    clusterer = WellClusterer(n_clusters=n_clusters, method="kmeans")
    clusterer.fit(summary)
    df_clustered = clusterer.assign_to_df(df, summary)

    train_df, val_df = split_wells(df_clustered)
    X_train, y_train = prepare_catboost_features(train_df, geo_priors)
    X_val, y_val = prepare_catboost_features(val_df, geo_priors)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    print("[trainer] Training cluster-aware CatBoost baseline...")
    model = train_catboost(X_train, y_train, X_val, y_val)
    save_catboost(model, name="catboost_tvt_clustered")

    val_preds = predict_catboost(model, X_val)
    mae = float(np.mean(np.abs(val_preds - y_val)))
    rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
    print(f"[trainer] Clustered Val MAE: {mae:.4f}")
    print(f"[trainer] Clustered Val RMSE: {rmse:.4f}")

    return {"mae": mae, "rmse": rmse, "n_features": X_train.shape[1], "n_clusters": n_clusters}


def predict_test(data_path: str, model_name: str = "catboost_tvt",
                 geo_priors: Optional[dict] = None) -> pd.DataFrame:
    """
    Generate predictions on test/evaluation data.
    Only predicts rows where TVT_input is NaN (the actual target zone).
    """
    from model.predictor import load_catboost

    df = load_competition_data(data_path)
    feat_df = engineer_sequence_features(df, geo_prior=geo_priors)
    feat_df = feat_df.select_dtypes(include=[np.number]).fillna(0)

    model = load_catboost(model_name)

    # Get model's expected features
    expected_cols = model.feature_names_
    X = feat_df.reindex(columns=expected_cols, fill_value=0)

    preds = predict_catboost(model, X)

    result = df[["well_id", "MD"]].copy() if "well_id" in df.columns else df[["MD"]].copy()
    result["TVT_predicted"] = preds

    out_path = MODEL_DIR / "predictions.csv"
    result.to_csv(out_path, index=False)
    print(f"[trainer] Saved predictions → {out_path}")
    return result
