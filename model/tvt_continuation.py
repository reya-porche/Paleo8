"""
geo_tvt/model/tvt_continuation.py
The core competition task: autoregressive TVT continuation.

The competition gives you:
  - TVT_input: known TVT up to some depth, then NaN
  - GR, MD, XYZ: complete throughout
  - Typewell GR + formation tops
  - Nearby well TVT histories

This module handles the NaN zone prediction using:
  1. Kalman smoother (physics-based, works with few wells)
  2. Geological-constrained autoregression (handles layer transitions)
  3. Ensemble from similar wells (transfer TVT patterns)
  4. Uncertainty-calibrated output
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Kalman Filter TVT Tracker ───────────────────────────────────────────────

class TVTKalmanTracker:
    """
    Kalman filter that tracks TVT along the well path.

    State: [tvt, tvt_velocity, tvt_acceleration]
    Observation: measured TVT (when available)

    This is powerful because:
      - It uses physical continuity constraints
      - It propagates uncertainty correctly
      - It can predict forward into the NaN zone using momentum
      - It can assimilate external geological priors as observation noise
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        obs_noise: float     = 0.1,
        prior_noise: float   = 0.5,   # noise when using geological prior as obs
    ):
        # State: [tvt, velocity, acceleration]
        self.n = 3
        self.F = np.array([   # state transition (constant acceleration model)
            [1, 1, 0.5],
            [0, 1, 1  ],
            [0, 0, 1  ],
        ], dtype=np.float64)
        self.H     = np.array([[1, 0, 0]], dtype=np.float64)  # observe TVT
        self.Q     = np.eye(3) * process_noise                # process noise
        self.Q[2, 2] *= 10   # acceleration changes more
        self.R_obs  = np.array([[obs_noise]])                  # observation noise
        self.R_prior = np.array([[prior_noise]])               # prior noise

        # State estimate
        self.x = np.zeros(3)   # [tvt, vel, acc]
        self.P = np.eye(3) * 1.0  # covariance

        self._initialized = False

    def initialize(self, tvt_value: float, velocity: float = 0.0) -> None:
        self.x = np.array([tvt_value, velocity, 0.0])
        self.P = np.diag([0.01, 0.1, 0.5])
        self._initialized = True

    def predict(self) -> tuple[float, float]:
        """Predict next state. Returns (tvt_pred, tvt_std)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(np.sqrt(self.P[0, 0]))

    def update(self, tvt_obs: float, use_prior: bool = False) -> None:
        """Assimilate a TVT observation."""
        R = self.R_prior if use_prior else self.R_obs
        y = tvt_obs - (self.H @ self.x)[0]
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P

    def run_smoother(
        self,
        tvt_series: np.ndarray,        # TVT values, NaN in prediction zone
        geological_priors: Optional[np.ndarray] = None,  # prior TVT per step
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman forward pass + RTS smoother on full series.

        Returns: (smoothed_tvt, uncertainty_per_step)
        """
        n = len(tvt_series)
        xs = np.zeros((n, 3))
        Ps = np.zeros((n, 3, 3))

        # Forward pass
        for i, obs in enumerate(tvt_series):
            tvt_pred, tvt_std = self.predict()

            if not self._initialized:
                if np.isfinite(obs):
                    self.initialize(obs)
                xs[i] = self.x
                Ps[i] = self.P
                continue

            if np.isfinite(obs):
                self.update(float(obs), use_prior=False)
            elif geological_priors is not None and np.isfinite(geological_priors[i]):
                self.update(float(geological_priors[i]), use_prior=True)

            xs[i] = self.x.copy()
            Ps[i] = self.P.copy()

        # RTS backward smoother
        xs_smooth = xs.copy()
        Ps_smooth = Ps.copy()
        for i in range(n - 2, -1, -1):
            P_pred = self.F @ Ps[i] @ self.F.T + self.Q
            G = Ps[i] @ self.F.T @ np.linalg.pinv(P_pred)
            xs_smooth[i] += G @ (xs_smooth[i + 1] - self.F @ xs[i])
            Ps_smooth[i] += G @ (Ps_smooth[i + 1] - P_pred) @ G.T

        tvt_smoothed = xs_smooth[:, 0]
        uncertainties = np.sqrt(np.clip(Ps_smooth[:, 0, 0], 0, None))
        return tvt_smoothed, uncertainties


# ─── Geological-Constrained Autoregression ───────────────────────────────────

def geological_autoregress(
    known_tvt: np.ndarray,            # TVT values in known zone
    known_gr: np.ndarray,             # GR in known zone
    pred_gr: np.ndarray,              # GR in prediction zone (fully known)
    continuity_prior: dict,           # from geological.py
    n_steps: int = 20,
    ar_order: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Autoregressive TVT prediction using GR-guided AR model.

    Key insight: TVT changes are driven by lithology changes.
    GR is a proxy for lithology → use GR change as an external regressor.

    AR model: TVT[t] = Σ a_i * TVT[t-i] + b * GR_change[t] + noise

    Returns: (predictions, uncertainties)
    """
    if len(known_tvt) < ar_order + 1:
        ar_order = max(2, len(known_tvt) - 1)

    # Fit AR model on known zone
    n = len(known_tvt)
    y = known_tvt[ar_order:]
    X_rows = []
    for i in range(ar_order, n):
        ar_terms = known_tvt[i - ar_order:i][::-1]
        gr_change = known_gr[i] - known_gr[i - 1] if i > 0 else 0.0
        X_rows.append(np.append(ar_terms, [gr_change, 1.0]))
    X = np.array(X_rows)

    # Ridge regression (stable for short series)
    ridge_lambda = 0.1
    coeffs = np.linalg.solve(X.T @ X + ridge_lambda * np.eye(X.shape[1]), X.T @ y)

    # Residual std (in-sample)
    resid_std = float(np.std(y - X @ coeffs))

    # Predict forward
    preds = []
    stds  = []
    tvt_buffer = list(known_tvt[-ar_order:])

    max_jump = continuity_prior.get("max_plausible_jump", 2.0)

    for step in range(n_steps):
        ar_terms  = np.array(tvt_buffer[-ar_order:][::-1])
        gr_change = pred_gr[step] - pred_gr[step - 1] if step > 0 else 0.0
        features  = np.append(ar_terms, [gr_change, 1.0])
        raw_pred  = float(coeffs @ features)

        # Geological plausibility constraint
        delta = raw_pred - tvt_buffer[-1]
        if abs(delta) > max_jump:
            raw_pred = tvt_buffer[-1] + np.sign(delta) * max_jump

        preds.append(raw_pred)
        tvt_buffer.append(raw_pred)

        # Uncertainty grows with steps (accumulated error)
        step_std = resid_std * np.sqrt(1 + step * 0.1)
        stds.append(float(step_std))

    return np.array(preds), np.array(stds)


# ─── Similar-Well TVT Transfer ───────────────────────────────────────────────

def transfer_tvt_from_similar_wells(
    query_tvt_known: np.ndarray,
    similar_well_tvts: list[np.ndarray],
    similarity_scores: list[float],
    n_pred_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensemble TVT continuation from K similar wells.

    For each similar well:
      1. Align TVT series at the observation horizon
      2. Extract their continuation pattern
      3. Blend by similarity weight

    Returns: (ensemble_prediction, uncertainty)
    """
    if not similar_well_tvts:
        return np.full(n_pred_steps, query_tvt_known[-1]), np.full(n_pred_steps, 1.0)

    anchor_tvt = float(query_tvt_known[-1])
    trend_diff = float(np.mean(np.diff(query_tvt_known[-5:]))) if len(query_tvt_known) >= 5 else 0.0

    weighted_preds = np.zeros(n_pred_steps)
    total_weight   = 0.0
    all_preds      = []

    for well_tvt, score in zip(similar_well_tvts, similarity_scores):
        if len(well_tvt) < n_pred_steps:
            continue

        # Normalize similar well TVT to start at same anchor
        # and preserve relative changes (not absolute values)
        well_anchor = float(well_tvt[0])
        continuation = well_tvt[:n_pred_steps] - well_anchor + anchor_tvt

        weighted_preds += score * continuation
        total_weight   += score
        all_preds.append(continuation)

    if total_weight < 1e-9:
        # Fallback: linear extrapolation
        steps = np.arange(1, n_pred_steps + 1)
        return anchor_tvt + trend_diff * steps, np.full(n_pred_steps, 0.5)

    ensemble = weighted_preds / total_weight

    # Uncertainty = weighted spread across similar wells
    if len(all_preds) > 1:
        stack = np.array(all_preds)
        weights = np.array(similarity_scores[:len(all_preds)])
        weights /= weights.sum()
        variance = np.average((stack - ensemble) ** 2, axis=0, weights=weights)
        uncertainty = np.sqrt(variance)
    else:
        uncertainty = np.full(n_pred_steps, 0.3)

    return ensemble, uncertainty


# ─── Full Continuation Pipeline ───────────────────────────────────────────────

def predict_tvt_continuation(
    well_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    similar_wells: list[pd.DataFrame],
    similarity_scores: list[float],
    continuity_prior: dict,
    tvt_col: str = "TVT_input",
    gr_col:  str = "GR",
    md_col:  str = "MD",
) -> pd.DataFrame:
    """
    Full TVT prediction pipeline for a single well.
    Combines Kalman smoother, AR model, and similar-well transfer.

    Returns: well_df with added columns:
      - TVT_pred_kalman
      - TVT_pred_ar
      - TVT_pred_transfer
      - TVT_pred_ensemble  ← primary prediction
      - TVT_uncertainty
    """
    out = well_df.copy()

    tvt = well_df[tvt_col].values
    gr  = well_df[gr_col].fillna(well_df[gr_col].median()).values

    known_mask = np.isfinite(tvt)
    pred_mask  = ~known_mask

    n_pred = int(pred_mask.sum())
    if n_pred == 0:
        out["TVT_pred_ensemble"] = tvt
        out["TVT_uncertainty"]   = 0.0
        return out

    known_tvt = tvt[known_mask]
    known_gr  = gr[known_mask]
    pred_gr   = gr[pred_mask]

    # ── Method 1: Kalman ─────────────────────────────────────────────────────
    tracker = TVTKalmanTracker(
        process_noise=continuity_prior.get("delta_tvt_std", 0.1) ** 2,
    )
    kal_preds, kal_std = tracker.run_smoother(tvt)

    # ── Method 2: AR + GR ────────────────────────────────────────────────────
    ar_preds, ar_std = geological_autoregress(
        known_tvt, known_gr, pred_gr, continuity_prior, n_steps=n_pred
    )

    # ── Method 3: Similar-well transfer ──────────────────────────────────────
    sim_tvt_series = [
        df[tvt_col].dropna().values for df in similar_wells
        if tvt_col in df.columns and df[tvt_col].notna().sum() >= n_pred
    ]
    transfer_preds, transfer_std = transfer_tvt_from_similar_wells(
        known_tvt, sim_tvt_series, similarity_scores[:len(sim_tvt_series)], n_pred
    )

    # ── Ensemble (weighted by method reliability) ────────────────────────────
    kal_weight = 0.4
    ar_weight  = 0.35
    tr_weight  = 0.25

    ensemble_preds = (
        kal_weight * kal_preds[pred_mask] +
        ar_weight  * ar_preds +
        tr_weight  * transfer_preds
    )
    ensemble_std = np.sqrt(
        kal_weight * kal_std[pred_mask]**2 +
        ar_weight  * ar_std**2 +
        tr_weight  * transfer_std**2
    )

    # Write back
    out["TVT_pred_kalman"]   = kal_preds
    out["TVT_pred_ar"]       = np.where(known_mask, tvt, np.concatenate([known_tvt[-1:].repeat(known_mask.sum()), ar_preds]))
    out["TVT_pred_ensemble"] = tvt.copy()
    out.loc[pred_mask, "TVT_pred_ensemble"] = ensemble_preds
    out["TVT_uncertainty"]   = 0.0
    out.loc[pred_mask, "TVT_uncertainty"] = ensemble_std

    return out
