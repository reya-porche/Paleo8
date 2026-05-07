"""
geo_tvt/evaluation/ablation.py
Ablation testing framework.

Answers the competition's most important question:
  "Does geological context actually improve TVT prediction?"

Runs systematic experiments:
  - Baseline (no geological features)
  - + Alignment features
  - + Formation geometry
  - + Cross-well representations
  - + Full system

Reports per-ablation MAE/RMSE and lift over baseline.
This becomes your pitch: "X% improvement from geological priors."
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR


@dataclass
class AblationResult:
    name:        str
    mae:         float
    rmse:        float
    mae_lift:    float  # % improvement over baseline
    rmse_lift:   float
    n_features:  int
    description: str = ""

    def __str__(self):
        sign = "+" if self.mae_lift > 0 else ""
        return (
            f"{self.name:<35} MAE={self.mae:.4f}  RMSE={self.rmse:.4f}  "
            f"Lift={sign}{self.mae_lift:.1f}%  Features={self.n_features}"
        )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))


class AblationStudy:
    """
    Runs a systematic ablation study over feature groups.

    Usage:
        study = AblationStudy(train_df, val_df, target_col="TVT")
        study.add_feature_group("baseline",      baseline_features)
        study.add_feature_group("+ alignment",   alignment_features)
        study.add_feature_group("+ geometry",    geometry_features)
        study.run()
        study.print_report()
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        target_col: str = "TVT",
        well_id_col: str = "well_id",
    ):
        self.train_df    = train_df
        self.val_df      = val_df
        self.target_col  = target_col
        self.well_id_col = well_id_col
        self.feature_groups: list[tuple] = []   # (name, feature_list, description)
        self.results: list[AblationResult] = []
        self._baseline_mae  = None
        self._baseline_rmse = None

    def add_feature_group(
        self,
        name: str,
        features: list[str],
        description: str = "",
        cumulative: bool = True,
    ) -> "AblationStudy":
        """
        Add a feature group to test.
        If cumulative=True, features from all previous groups are included.
        """
        self.feature_groups.append((name, features, description, cumulative))
        return self

    def _fit_predict(self, X_train, y_train, X_val) -> np.ndarray:
        """Train CatBoost and predict."""
        try:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=300, learning_rate=0.05,
                                      depth=6, verbose=0, random_seed=42)
            model.fit(X_train, y_train)
            return model.predict(X_val)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                               learning_rate=0.05, random_state=42)
            model.fit(X_train, y_train)
            return model.predict(X_val)

    def _get_split(self, features: list[str]):
        """Get train/val splits for a feature list."""
        available_train = [f for f in features if f in self.train_df.columns]
        available_val   = [f for f in features if f in self.val_df.columns]
        common = list(set(available_train) & set(available_val))

        if not common:
            return None, None, None, None

        mask_train = self.train_df[self.target_col].notna()
        mask_val   = self.val_df[self.target_col].notna()

        X_train = self.train_df[mask_train][common].select_dtypes(include=[np.number]).fillna(0)
        y_train = self.train_df[mask_train][self.target_col].values
        X_val   = self.val_df[mask_val][common].select_dtypes(include=[np.number]).fillna(0)
        y_val   = self.val_df[mask_val][self.target_col].values

        return X_train, y_train, X_val, y_val

    def run(self) -> list[AblationResult]:
        """Run all ablation experiments."""
        cumulative_features: list[str] = []
        results = []

        for name, features, description, cumulative in self.feature_groups:
            if cumulative:
                current_features = cumulative_features + features
            else:
                current_features = features

            X_train, y_train, X_val, y_val = self._get_split(current_features)
            if X_train is None:
                print(f"[ablation] Skipping '{name}': no matching features found")
                continue

            preds = self._fit_predict(X_train, y_train, X_val)
            m = mae(y_val, preds)
            r = rmse(y_val, preds)

            if self._baseline_mae is None:
                self._baseline_mae  = m
                self._baseline_rmse = r
                mae_lift  = 0.0
                rmse_lift = 0.0
            else:
                mae_lift  = (self._baseline_mae  - m) / self._baseline_mae  * 100
                rmse_lift = (self._baseline_rmse - r) / self._baseline_rmse * 100

            result = AblationResult(
                name=name, mae=m, rmse=r,
                mae_lift=mae_lift, rmse_lift=rmse_lift,
                n_features=X_train.shape[1],
                description=description,
            )
            results.append(result)
            print(f"  {result}")

            if cumulative:
                cumulative_features = current_features

        self.results = results
        return results

    def print_report(self) -> None:
        print("\n" + "=" * 90)
        print("ABLATION STUDY RESULTS")
        print("=" * 90)
        print(f"{'Configuration':<35} {'MAE':>8} {'RMSE':>8} {'MAE Lift':>10} {'Features':>10}")
        print("-" * 90)
        for r in self.results:
            sign = "+" if r.mae_lift > 0 else ""
            print(
                f"{r.name:<35} {r.mae:>8.4f} {r.rmse:>8.4f} "
                f"{sign}{r.mae_lift:>8.1f}% {r.n_features:>10}"
            )
        print("=" * 90)

        if len(self.results) >= 2:
            best = max(self.results, key=lambda r: r.mae_lift)
            print(f"\n→ Best configuration: '{best.name}' ({best.mae_lift:+.1f}% MAE improvement)")
            print(f"→ Geological features provide {best.mae_lift:.1f}% lift over baseline.")
            print("\n  This is your competition pitch number.")

    def save_report(self, path: Optional[Path] = None) -> None:
        path = path or (MODEL_DIR / "ablation_report.csv")
        df = pd.DataFrame([
            {"name": r.name, "mae": r.mae, "rmse": r.rmse,
             "mae_lift_pct": r.mae_lift, "rmse_lift_pct": r.rmse_lift,
             "n_features": r.n_features}
            for r in self.results
        ])
        df.to_csv(path, index=False)
        print(f"[ablation] Saved report → {path}")


# ─── Quick Ablation Builder ───────────────────────────────────────────────────

def run_standard_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    baseline_cols: list[str],
    alignment_cols: list[str],
    geometry_cols: list[str],
    repr_cols: list[str],
    target_col: str = "TVT",
) -> AblationStudy:
    """
    Run the standard four-stage ablation for this competition.
    
    baseline_cols:   sequence features only (GR, MD, lags, rolling)
    alignment_cols:  typewell alignment features
    geometry_cols:   formation surface geometry features
    repr_cols:       cross-well representation features
    """
    study = AblationStudy(train_df, val_df, target_col)
    study.add_feature_group(
        "1. Baseline (sequence only)",
        baseline_cols,
        "GR + MD + lags + rolling stats",
    )
    study.add_feature_group(
        "2. + Typewell alignment",
        alignment_cols,
        "DTW-aligned depth, GR deviation, formation proximity",
    )
    study.add_feature_group(
        "3. + Formation geometry",
        geometry_cols,
        "Predicted surface depths, dip correction, thickness",
    )
    study.add_feature_group(
        "4. + Cross-well representations",
        repr_cols,
        "GR embedding similarity, similar-well TVT patterns",
    )
    return study
