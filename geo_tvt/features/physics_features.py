"""
geo_tvt/features/physics_features.py
Physics-derived features for geological dynamics.

The model should learn to predict ΔTVT, not absolute TVT.
These features give the model the physical quantities that drive TVT change:

  - Structural gradients: d(Z)/d(MD), d(GR)/d(MD), d(TVT)/d(MD)
  - Trajectory curvature: second derivatives, turning rate
  - Formation dip proxy: spatial TVT gradient across the lateral
  - State-space approximations: geological state from observable signals

This is what separates "ML on logs" from "physics-informed ML."
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Optional


def build_physics_features(
    df: pd.DataFrame,
    gr_col:  str = "GR",
    md_col:  str = "MD",
    x_col:   str = "X",
    y_col:   str = "Y",
    z_col:   str = "Z",
    tvt_col: str = "TVT_input",
) -> pd.DataFrame:
    """
    Compute physics-derived features for a single well's dataframe.

    Key idea: instead of predicting TVT directly, the model sees the
    physical rates of change that will determine TVT movement.
    """
    feat = df.copy()
    n = len(feat)
    if n < 3:
        return feat

    # Safe column access
    def col(name):
        if name in feat.columns:
            return feat[name].interpolate(limit=5).fillna(0).values.astype(np.float64)
        return np.zeros(n)

    gr  = col(gr_col)
    md  = col(md_col)
    x   = col(x_col)
    y   = col(y_col)
    z   = col(z_col)
    tvt_raw = feat[tvt_col].values if tvt_col in feat.columns else np.zeros(n)
    tvt = np.where(np.isfinite(tvt_raw), tvt_raw, np.nan)

    # MD step (arc length increment)
    md_step = np.diff(md, prepend=md[0])
    md_step = np.where(md_step > 0, md_step, np.nanmedian(md_step[md_step > 0]) or 1.0)

    # ── Structural gradients ─────────────────────────────────────────────────

    # d(Z)/d(MD) — inclination (how steeply the well is diving)
    feat["dz_dmd"] = _safe_gradient(z, md_step)
    # d(GR)/d(MD) — lithology transition rate
    feat["dgr_dmd"] = _safe_gradient(gr, md_step)
    # d²(Z)/d(MD²) — change in inclination = structural curvature
    feat["d2z_dmd2"] = _safe_gradient(feat["dz_dmd"].values, md_step)
    # d²(GR)/d(MD²) — rate of lithology transition change
    feat["d2gr_dmd2"] = _safe_gradient(feat["dgr_dmd"].values, md_step)

    # ── Trajectory curvature ─────────────────────────────────────────────────

    # 3D position derivatives
    dx = _safe_gradient(x, md_step)
    dy = _safe_gradient(y, md_step)
    dz = _safe_gradient(z, md_step)
    feat["dx_dmd"] = dx
    feat["dy_dmd"] = dy

    # Speed along 3D path (should be ~1 for properly parameterized MD)
    feat["path_speed"] = np.sqrt(dx**2 + dy**2 + dz**2)

    # Turning rate in horizontal plane (azimuth change)
    horiz_speed = np.sqrt(dx**2 + dy**2) + 1e-9
    azimuth = np.degrees(np.arctan2(dy, dx))
    feat["azimuth_rate"] = _safe_gradient(azimuth, md_step)
    feat["azimuth_rate_abs"] = np.abs(feat["azimuth_rate"].values)

    # 3D curvature magnitude
    ddx = _safe_gradient(dx, md_step)
    ddy = _safe_gradient(dy, md_step)
    ddz = _safe_gradient(dz, md_step)
    cross_mag = np.sqrt(
        (dy * ddz - dz * ddy)**2 +
        (dz * ddx - dx * ddz)**2 +
        (dx * ddy - dy * ddx)**2
    )
    feat["curvature_3d"] = np.clip(cross_mag / (feat["path_speed"].values**3 + 1e-9), 0, 1.0)

    # Dogleg severity proxy (common drilling metric)
    feat["dogleg_proxy"] = np.sqrt(
        feat["d2z_dmd2"].values**2 + feat["azimuth_rate"].values**2
    )

    # ── TVT physical derivatives ─────────────────────────────────────────────

    tvt_filled = pd.Series(tvt).interpolate(limit=3).values
    # d(TVT)/d(MD) — TVT change per foot drilled
    feat["dtvt_dmd"]    = _safe_gradient(tvt_filled, md_step)
    # d²(TVT)/d(MD²) — curvature of TVT trajectory
    feat["d2tvt_dmd2"]  = _safe_gradient(feat["dtvt_dmd"].values, md_step)
    # Smooth TVT rate (less noisy)
    feat["dtvt_smooth"] = _savgol_derivative(tvt_filled, window=11, polyorder=2)

    # ── Formation dip proxy ──────────────────────────────────────────────────

    # True dip = how much TVT changes per unit of horizontal movement
    horiz_dist = np.sqrt(x**2 + y**2)
    dhoriz = _safe_gradient(horiz_dist, md_step)
    dhoriz_safe = np.where(np.abs(dhoriz) > 0.01, dhoriz, np.sign(dhoriz) * 0.01)
    feat["formation_dip_proxy"] = feat["dtvt_dmd"].values / (dhoriz_safe + 1e-9)
    feat["formation_dip_proxy"] = np.clip(feat["formation_dip_proxy"].values, -10.0, 10.0)

    # ── Physical state indicators ────────────────────────────────────────────

    # GR / Z coupling — high GR + increasing Z = shaly section going deeper
    feat["gr_z_coupling"] = feat["dgr_dmd"].values * feat["dz_dmd"].values

    # "Geological state change" signal — weighted combination of all rate signals
    feat["geo_state_change"] = (
        0.4 * np.abs(feat["dgr_dmd"].values) / (np.nanstd(feat["dgr_dmd"].values) + 1e-9) +
        0.3 * np.abs(feat["d2z_dmd2"].values) / (np.nanstd(feat["d2z_dmd2"].values) + 1e-9) +
        0.3 * np.abs(feat["dtvt_dmd"].values) / (np.nanstd(feat["dtvt_dmd"].values) + 1e-9)
    )

    # Entry into formation transition zone
    feat["transition_signal"] = (
        np.abs(feat["dgr_dmd"].values) > np.nanpercentile(np.abs(feat["dgr_dmd"].values), 75)
    ).astype(float)

    # ── Savgol smoothed physics features (noise-reduced) ────────────────────
    feat["gr_savgol"]  = _savgol_smooth(gr)
    feat["z_savgol"]   = _savgol_smooth(z)
    feat["gr_savgol_d1"] = _savgol_derivative(gr, window=11, polyorder=3)
    feat["z_savgol_d1"]  = _savgol_derivative(z, window=11, polyorder=3)

    return feat.fillna(0)


# ─── Numerical utilities ─────────────────────────────────────────────────────

def _safe_gradient(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """dy/dx with safe handling of non-uniform spacing."""
    dy = np.diff(y, prepend=y[0])
    dx_safe = np.where(np.abs(dx) > 1e-9, dx, 1.0)
    return dy / dx_safe


def _savgol_smooth(y: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing. Falls back to uniform filter for short series."""
    window = min(window, len(y) - (1 if len(y) % 2 == 0 else 0))
    if window < polyorder + 1 or window < 3:
        return y.copy()
    if window % 2 == 0:
        window -= 1
    try:
        return savgol_filter(y, window_length=window, polyorder=polyorder)
    except Exception:
        return y.copy()


def _savgol_derivative(y: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay derivative. Numerically cleaner than finite differences."""
    window = min(window, len(y) - (1 if len(y) % 2 == 0 else 0))
    if window < polyorder + 1 or window < 3:
        return np.gradient(y)
    if window % 2 == 0:
        window -= 1
    try:
        return savgol_filter(y, window_length=window, polyorder=polyorder, deriv=1)
    except Exception:
        return np.gradient(y)


def build_physics_features_for_dataset(
    df: pd.DataFrame,
    well_col: str = "well_id",
    **kwargs,
) -> pd.DataFrame:
    """Apply per-well physics feature engineering to the full dataset."""
    if well_col not in df.columns:
        return build_physics_features(df, **kwargs)
    parts = []
    for _, grp in df.groupby(well_col, sort=False):
        parts.append(build_physics_features(grp.copy(), **kwargs))
    return pd.concat(parts, ignore_index=True)
