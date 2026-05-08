"""
geo_tvt/features/sequence_features.py
Comprehensive sequence feature engineering.

The baseline used: MD, X, Y, Z, GR, TVT_input — 6 raw columns.
This module produces 80+ engineered features encoding geological dynamics.

Key insight: TVT continuation is driven by:
  1. Recent TVT trajectory (momentum + curvature)
  2. GR behavior (lithology transitions)
  3. Well trajectory geometry (physical drilling path)
  4. Distance from the NaN boundary (uncertainty grows)
  5. Pre-NaN zone slope — the single most powerful feature
"""

import numpy as np
import pandas as pd
from typing import Optional


def build_all_sequence_features(
    df: pd.DataFrame,
    gr_col:   str = "GR",
    md_col:   str = "MD",
    x_col:    str = "X",
    y_col:    str = "Y",
    z_col:    str = "Z",
    tvt_col:  str = "TVT_input",
    well_col: str = "well_id",
) -> pd.DataFrame:
    """
    Build all sequence features.
    Apply per-well via groupby.apply() — never across wells.
    """
    feat = df.copy()

    # ── GR features ──────────────────────────────────────────────────────────
    gr = feat[gr_col].interpolate(limit=5).fillna(feat[gr_col].median()) \
         if gr_col in feat.columns else pd.Series(0.0, index=feat.index)

    for w in [3, 5, 10, 20, 50]:
        feat[f"gr_rmean_{w}"] = gr.rolling(w, min_periods=1).mean()
        feat[f"gr_rstd_{w}"]  = gr.rolling(w, min_periods=1).std().fillna(0)
        feat[f"gr_rmin_{w}"]  = gr.rolling(w, min_periods=1).min()
        feat[f"gr_rmax_{w}"]  = gr.rolling(w, min_periods=1).max()

    for w in [10, 20, 50]:
        feat[f"gr_range_{w}"] = (
            gr.rolling(w, min_periods=1).max() -
            gr.rolling(w, min_periods=1).min()
        )

    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"gr_lag{lag}"]  = gr.shift(lag)
        feat[f"gr_lead{lag}"] = gr.shift(-lag)

    feat["gr_d1"]          = gr.diff().fillna(0)
    feat["gr_d2"]          = gr.diff().diff().fillna(0)
    feat["gr_d1_abs"]      = feat["gr_d1"].abs()
    feat["gr_grad_smooth"] = gr.rolling(5, min_periods=1).mean().diff().fillna(0)
    feat["gr_sand_proxy"]  = (gr < 60).astype(float)
    feat["gr_shale_proxy"] = (gr > 100).astype(float)

    # ── Trajectory features ───────────────────────────────────────────────────
    if md_col in feat.columns:
        feat["md_step"] = feat[md_col].diff().fillna(feat[md_col].diff().median())
        md_min = feat[md_col].min()
        md_max = feat[md_col].max()
        feat["md_frac"] = (feat[md_col] - md_min) / max(1.0, md_max - md_min)

    if x_col in feat.columns and y_col in feat.columns:
        x, y = feat[x_col], feat[y_col]
        feat["horiz_dist"]    = np.sqrt(x**2 + y**2)
        feat["x_d1"]          = x.diff().fillna(0)
        feat["y_d1"]          = y.diff().fillna(0)
        feat["horiz_speed"]   = np.sqrt(feat["x_d1"]**2 + feat["y_d1"]**2)
        feat["horiz_dir_deg"] = np.degrees(np.arctan2(feat["y_d1"], feat["x_d1"]))

    if z_col in feat.columns:
        z = feat[z_col]
        feat["z_d1"]      = z.diff().fillna(0)
        feat["z_d2"]      = z.diff().diff().fillna(0)
        feat["z_d1_abs"]  = feat["z_d1"].abs()
        feat["z_rmean_10"] = z.rolling(10, min_periods=1).mean()
        feat["z_deviation"] = z - feat["z_rmean_10"]
        for w in [5, 10, 20]:
            feat[f"z_curve_{w}"] = feat["z_d1"].rolling(w, min_periods=1).std().fillna(0)

    # ── TVT continuation features — most important ────────────────────────────
    if tvt_col in feat.columns:
        tvt = feat[tvt_col]
        tvt_ffill = tvt.ffill()
        feat["tvt_ffill"] = tvt_ffill

        last_known_idx = tvt.last_valid_index()
        feat["last_known_tvt"] = float(tvt.loc[last_known_idx]) if last_known_idx is not None else tvt_ffill
        feat["dist_from_last_known"] = _distance_from_last_known(tvt)
        feat["in_nan_zone"] = tvt.isna().astype(float)

        tvt_filled = tvt.interpolate(limit=3)
        feat["tvt_d1"] = tvt_filled.diff().fillna(0)
        feat["tvt_d2"] = tvt_filled.diff().diff().fillna(0)

        # Pre-NaN slope features — the runway
        for window in [5, 10, 20, 50, 100, 200]:
            feat[f"tvt_pre_slope_{window}"] = _pre_nan_slope(tvt, window)

        # Linear extrapolation: last_known + slope × distance_into_NaN_zone
        # This gives the model a direct "free answer" for linear continuation.
        dist = feat["dist_from_last_known"]
        for window in [5, 10, 20, 50]:
            feat[f"tvt_extrap_{window}"] = (
                feat["last_known_tvt"] + feat[f"tvt_pre_slope_{window}"] * dist
            )

        for w in [5, 10, 20, 50]:
            feat[f"tvt_rmean_{w}"]  = tvt_ffill.rolling(w, min_periods=1).mean()
            feat[f"tvt_rstd_{w}"]   = tvt_ffill.rolling(w, min_periods=1).std().fillna(0)

        for lag in [1, 2, 3, 5, 10, 20]:
            feat[f"tvt_lag{lag}"] = tvt_ffill.shift(lag)

        # Anchor-zone statistics — well-level constants, critical for cross-well generalization
        anchor_mask = tvt.notna()
        anchor_tvt = tvt[anchor_mask]
        nan_count = float(tvt.isna().sum())
        anchor_count = float(anchor_mask.sum())

        if len(anchor_tvt) >= 2:
            x_anc = np.arange(len(anchor_tvt), dtype=float)
            anchor_global_slope = float(np.polyfit(x_anc, anchor_tvt.values, 1)[0])
        else:
            anchor_global_slope = 0.0

        feat["anchor_tvt_mean"]   = float(anchor_tvt.mean())   if len(anchor_tvt) > 0 else 0.0
        feat["anchor_tvt_last"]   = float(anchor_tvt.iloc[-1]) if len(anchor_tvt) > 0 else 0.0
        feat["anchor_tvt_range"]  = float(anchor_tvt.max() - anchor_tvt.min()) if len(anchor_tvt) > 1 else 0.0
        feat["anchor_global_slope"] = anchor_global_slope
        feat["anchor_length"]     = anchor_count
        feat["nan_length"]        = nan_count
        feat["tvt_known_frac"]    = anchor_count / max(1.0, anchor_count + nan_count)
        feat["nan_to_anchor_ratio"] = nan_count / max(1.0, anchor_count)

        # GR context at the anchor-NaN boundary
        if gr_col in feat.columns:
            nan_start = int(tvt.isna().values.argmax()) if tvt.isna().any() else len(tvt)
            if nan_start > 0:
                boundary_gr = gr.iloc[max(0, nan_start - 20):nan_start]
                feat["gr_at_boundary"]     = float(boundary_gr.mean())
                feat["gr_std_at_boundary"] = float(boundary_gr.std()) if len(boundary_gr) > 1 else 0.0
            else:
                feat["gr_at_boundary"]     = float(gr.mean())
                feat["gr_std_at_boundary"] = float(gr.std()) if len(gr) > 1 else 0.0
            feat["gr_vs_pre_nan"] = gr - _pre_nan_mean(gr, tvt)

        if z_col in feat.columns:
            feat["z_vs_pre_nan"] = feat[z_col] - _pre_nan_mean(feat[z_col], tvt)

    return feat.fillna(0)


def build_features_for_dataset(
    df: pd.DataFrame,
    well_col: str = "well_id",
    **kwargs,
) -> pd.DataFrame:
    """
    Apply build_all_sequence_features per well.
    Use this on the full dataset instead of calling the function directly.
    """
    if well_col not in df.columns:
        return build_all_sequence_features(df, **kwargs)

    parts = []
    for _, grp in df.groupby(well_col, sort=False):
        parts.append(build_all_sequence_features(grp.copy(), **kwargs))
    return pd.concat(parts, ignore_index=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _distance_from_last_known(tvt: pd.Series) -> pd.Series:
    dist = pd.Series(0.0, index=tvt.index)
    counter = 0
    for i in range(len(tvt)):
        if pd.isna(tvt.iloc[i]):
            counter += 1
        else:
            counter = 0
        dist.iloc[i] = float(counter)
    return dist


def _pre_nan_slope(tvt: pd.Series, window_size: int) -> pd.Series:
    """Linear slope of TVT in the last window_size known rows, broadcast into NaN zone."""
    result = pd.Series(0.0, index=tvt.index)
    known = tvt.dropna()
    if len(known) < 2:
        return result
    window_vals = known.iloc[-min(window_size, len(known)):]
    if len(window_vals) < 2:
        return result
    x = np.arange(len(window_vals), dtype=float)
    slope = float(np.polyfit(x, window_vals.values, 1)[0])
    result[tvt.isna()] = slope
    return result


def _pre_nan_mean(signal: pd.Series, tvt: pd.Series, window: int = 20) -> float:
    known_signal = signal[tvt.notna()]
    if len(known_signal) == 0:
        return float(signal.mean())
    return float(known_signal.iloc[-min(window, len(known_signal)):].mean())


def get_feature_groups(df: pd.DataFrame) -> dict:
    """Return named column groups for ablation. Call after feature building."""
    c = list(df.columns)
    return {
        # TVT_input excluded from baseline — it becomes 0 in the NaN zone (misleading).
        # Use tvt_ffill (last-known carry-forward) instead.
        "baseline":         ["GR", "MD", "X", "Y", "Z"],
        "gr_rolling":       [x for x in c if x.startswith("gr_r")],
        "gr_derivatives":   [x for x in c if x.startswith("gr_d") or "gr_grad" in x],
        "gr_lags":          [x for x in c if "gr_lag" in x or "gr_lead" in x],
        "gr_proxies":       [x for x in c if "proxy" in x],
        "trajectory":       [x for x in c if any(k in x for k in
                              ["horiz", "x_d1", "y_d1", "z_d1", "z_d2", "z_curve",
                               "z_rm", "z_dev", "md_step", "md_frac"])],
        "tvt_continuation": [x for x in c if "pre_slope" in x or "tvt_extrap" in x or x in
                              ["last_known_tvt", "dist_from_last_known", "tvt_ffill",
                               "in_nan_zone", "gr_vs_pre_nan", "z_vs_pre_nan"]],
        "tvt_history":      [x for x in c if x.startswith("tvt_r") or x.startswith("tvt_lag")
                              or x in ["tvt_d1", "tvt_d2"]],
        "anchor_stats":     [x for x in c if x in [
                              "anchor_tvt_mean", "anchor_tvt_last", "anchor_tvt_range",
                              "anchor_global_slope", "anchor_length", "nan_length",
                              "tvt_known_frac", "nan_to_anchor_ratio",
                              "gr_at_boundary", "gr_std_at_boundary"]],
    }
