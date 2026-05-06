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
from config import MODEL_DIR
from model.predictor import (
    engineer_sequence_features,
    train_catboost, predict_catboost, save_catboost,
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
