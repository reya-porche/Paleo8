"""
geo_tvt/alignment/typewell_matcher.py
The most important module for the anonymized-coordinate case.

The typewell is the key geological reference for each horizontal well.
This module answers:
  "Where in the typewell's stratigraphy is each point along the lateral?"

Methods:
  1. DTW (Dynamic Time Warping) alignment of GR sequences
  2. Cross-correlation alignment (fast, good for first pass)
  3. Stratigraphic similarity scoring
  4. Nearest typewell selection (when multiple typewells exist)

The output is a depth mapping: each MD point on the lateral
gets a corresponding TVD/depth-in-typewell estimate.
This becomes a powerful feature for TVT prediction.
"""

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks
from scipy.ndimage import uniform_filter1d
from typing import Optional
from pathlib import Path


# ─── Signal Preprocessing ────────────────────────────────────────────────────

def normalize_gr(gr: np.ndarray, clip_pct: float = 99.0) -> np.ndarray:
    """Normalize GR to [0, 1] with outlier clipping."""
    gr = np.array(gr, dtype=np.float64)
    valid = gr[np.isfinite(gr)]
    if len(valid) == 0:
        return np.zeros_like(gr)
    lo, hi = np.nanpercentile(gr, 1), np.nanpercentile(gr, clip_pct)
    if hi == lo:
        return np.zeros_like(gr)
    return np.clip((gr - lo) / (hi - lo), 0.0, 1.0)


def smooth_gr(gr: np.ndarray, window: int = 5) -> np.ndarray:
    """Uniform smoothing to reduce noise before alignment."""
    return uniform_filter1d(gr, size=window, mode="nearest")


def resample_to_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Resample signal to target length using linear interpolation."""
    if len(signal) == target_len:
        return signal
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, signal)


# ─── DTW Alignment ────────────────────────────────────────────────────────────

def dtw_distance(s1: np.ndarray, s2: np.ndarray,
                 window: Optional[int] = None) -> tuple[float, np.ndarray]:
    """
    Dynamic Time Warping distance between two sequences.
    Returns (distance, warping_path).

    window: Sakoe-Chiba band width (None = no constraint).
    """
    n, m = len(s1), len(s2)
    w = max(window or max(n, m), abs(n - m))

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end   = min(m, i + w) + 1
        for j in range(j_start, j_end):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    # Traceback
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        options = []
        if i > 0 and j > 0:
            options.append((dtw_matrix[i-1, j-1], i-1, j-1))
        if i > 0:
            options.append((dtw_matrix[i-1, j],   i-1, j))
        if j > 0:
            options.append((dtw_matrix[i, j-1],   i,   j-1))
        _, i, j = min(options)

    path.reverse()
    dist = dtw_matrix[n, m] / (n + m)
    return dist, np.array(path)


def align_lateral_to_typewell(
    lateral_gr: np.ndarray,
    typewell_gr: np.ndarray,
    lateral_md: np.ndarray,
    typewell_depth: np.ndarray,
    smooth_window: int = 7,
    dtw_window: Optional[int] = None,
) -> dict:
    """
    Align a lateral well's GR log to the typewell GR log using DTW.

    Returns:
      - depth_mapping: for each lateral MD index, the corresponding
                       typewell depth (TVD proxy)
      - alignment_score: 0-1, higher = better geological match
      - warping_path: raw DTW alignment indices
    """
    # Preprocess
    lat_proc  = smooth_gr(normalize_gr(lateral_gr),  smooth_window)
    tw_proc   = smooth_gr(normalize_gr(typewell_gr), smooth_window)

    # Resample lateral to typewell length for fair comparison
    lat_resampled = resample_to_length(lat_proc, len(tw_proc))

    dist, path = dtw_distance(lat_resampled, tw_proc, window=dtw_window)

    # Build depth mapping
    # For each lateral MD position, find the corresponding typewell depth
    lat_indices = np.arange(len(lateral_gr))
    # Map from resampled indices back to original
    scale = len(lateral_gr) / len(lat_resampled)
    depth_mapping = np.interp(
        lat_indices,
        path[:, 0] * scale,
        typewell_depth[np.clip(path[:, 1], 0, len(typewell_depth) - 1)],
    )

    # Alignment score: lower DTW distance = better match
    alignment_score = float(np.exp(-dist))

    return {
        "depth_mapping":    depth_mapping,
        "alignment_score":  alignment_score,
        "dtw_distance":     float(dist),
        "warping_path":     path,
    }


# ─── Cross-Correlation Alignment ─────────────────────────────────────────────

def xcorr_align(lateral_gr: np.ndarray,
                typewell_gr: np.ndarray,
                typewell_depth: np.ndarray) -> dict:
    """
    Fast cross-correlation alignment.
    Good first pass; use DTW for refinement.

    Returns best lag (in typewell depth units) and correlation score.
    """
    lat_norm = normalize_gr(smooth_gr(lateral_gr))
    tw_norm  = normalize_gr(smooth_gr(typewell_gr))

    corr = correlate(tw_norm, lat_norm, mode="full")
    lags = np.arange(-(len(lat_norm) - 1), len(tw_norm))

    best_lag_idx = np.argmax(np.abs(corr))
    best_lag     = lags[best_lag_idx]
    corr_score   = float(corr[best_lag_idx] / (len(lat_norm) * np.std(tw_norm) * np.std(lat_norm) + 1e-9))

    # Translate lag to depth offset
    depth_step    = float(np.median(np.diff(typewell_depth))) if len(typewell_depth) > 1 else 1.0
    depth_offset  = best_lag * depth_step

    return {
        "depth_offset_ft": depth_offset,
        "correlation":     corr_score,
        "best_lag_steps":  int(best_lag),
    }


# ─── Stratigraphic Similarity ─────────────────────────────────────────────────

def gr_stratigraphic_similarity(gr_a: np.ndarray, gr_b: np.ndarray) -> float:
    """
    Compare two GR sequences for stratigraphic similarity.
    Uses normalized cross-correlation on common length.
    Returns 0-1 similarity score.
    """
    min_len = min(len(gr_a), len(gr_b))
    a = normalize_gr(gr_a[:min_len])
    b = normalize_gr(gr_b[:min_len])
    denom = (np.std(a) * np.std(b) * min_len)
    if denom < 1e-9:
        return 0.0
    return float(np.clip(np.correlate(a - a.mean(), b - b.mean())[0] / denom, 0.0, 1.0))


# ─── Multi-Typewell Selection ─────────────────────────────────────────────────

def select_best_typewell(
    lateral_gr: np.ndarray,
    typewells: dict[str, np.ndarray],
) -> tuple[str, float]:
    """
    When multiple typewells are available, pick the best match.

    typewells: {well_id: gr_array}
    Returns: (best_well_id, similarity_score)
    """
    scores = {}
    for well_id, tw_gr in typewells.items():
        score = gr_stratigraphic_similarity(lateral_gr, tw_gr)
        scores[well_id] = score

    best_id    = max(scores, key=scores.get)
    best_score = scores[best_id]
    return best_id, best_score


# ─── Feature Generation from Alignment ───────────────────────────────────────

def alignment_features(
    lateral_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    gr_col: str     = "GR",
    md_col: str     = "MD",
    depth_col: str  = "TVD",
) -> pd.DataFrame:
    """
    Generate alignment-based features for each row of the lateral.

    Adds to lateral_df:
      - aligned_depth:    estimated typewell depth position
      - alignment_score:  per-well DTW alignment quality
      - gr_deviation:     difference between lateral GR and expected GR at aligned position
      - formation_proximity: distance to nearest typewell formation top
    """
    lat_gr  = lateral_df[gr_col].fillna(lateral_df[gr_col].median()).values
    tw_gr   = typewell_df[gr_col].fillna(typewell_df[gr_col].median()).values
    tw_dep  = typewell_df[depth_col].values if depth_col in typewell_df.columns \
              else np.arange(len(tw_gr), dtype=float)
    lat_md  = lateral_df[md_col].values

    result = align_lateral_to_typewell(lat_gr, tw_gr, lat_md, tw_dep)

    out = lateral_df.copy()
    out["aligned_depth"]    = result["depth_mapping"]
    out["alignment_score"]  = result["alignment_score"]

    # GR deviation from typewell expectation at aligned depth
    tw_gr_interp = np.interp(
        result["depth_mapping"],
        tw_dep,
        normalize_gr(tw_gr),
    )
    out["gr_deviation"] = normalize_gr(lat_gr) - tw_gr_interp

    # Formation proximity
    if "formation_top" in typewell_df.columns:
        fm_tops = typewell_df["formation_top"].dropna().values
        if len(fm_tops) > 0:
            out["formation_proximity"] = out["aligned_depth"].apply(
                lambda d: float(np.min(np.abs(d - fm_tops)))
            )

    return out
