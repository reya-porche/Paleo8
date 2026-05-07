"""
geo_tvt/alignment/window_typewell.py
Top-K typewell window matching.

Instead of one global DTW alignment, for each point along the lateral:
  1. Extract a GR window around that point
  2. Slide that window across the typewell
  3. Find the top 5 matching positions
  4. Return the TVT values at those typewell positions as features

This gives the model:
  - typewell_best_match_tvt
  - typewell_match_score
  - typewell_second_best_tvt
  - typewell_match_spread (disagreement between top matches)
  - typewell_confidence

Much more powerful than a single alignment.
"""

import numpy as np
import pandas as pd
from scipy.signal import correlate
from typing import Optional


def sliding_window_match(
    lateral_gr:   np.ndarray,
    typewell_gr:  np.ndarray,
    typewell_tvt: np.ndarray,
    window_size:  int = 30,
    top_k:        int = 5,
    step:         int = 1,
) -> list[dict]:
    """
    For each position in lateral_gr, find the top-K matching
    positions in typewell_gr using a sliding correlation window.

    Returns a list of dicts, one per lateral position:
      {
        "best_tvt":     float,
        "best_score":   float,
        "top_k_tvts":   [float x K],
        "top_k_scores": [float x K],
        "tvt_spread":   float,
        "confidence":   float,
      }
    """
    n_lat  = len(lateral_gr)
    n_tw   = len(typewell_gr)
    half_w = window_size // 2

    # Normalize
    lat_norm = _normalize(lateral_gr)
    tw_norm  = _normalize(typewell_gr)
    tvt_filled = pd.Series(typewell_tvt).interpolate(limit=5).fillna(
        np.nanmean(typewell_tvt)
    ).values

    results = []

    for i in range(n_lat):
        # Extract lateral window around position i
        i_start = max(0, i - half_w)
        i_end   = min(n_lat, i + half_w)
        lat_win = lat_norm[i_start:i_end]
        win_len = len(lat_win)

        if win_len < 5:
            results.append(_empty_result(top_k))
            continue

        # Slide across typewell
        scores = []
        for j in range(0, n_tw - win_len + 1, step):
            tw_win = tw_norm[j: j + win_len]
            corr = _window_corr(lat_win, tw_win)
            scores.append((corr, j, j + win_len // 2))  # (score, start, center)

        if not scores:
            results.append(_empty_result(top_k))
            continue

        # Sort by score (descending)
        scores.sort(key=lambda x: -x[0])

        # Top-K unique positions (suppress near-duplicates)
        top_matches = _suppress_near_duplicates(scores, window_size, top_k)

        tvt_matches = []
        score_matches = []
        for score, _, center in top_matches[:top_k]:
            center = int(np.clip(center, 0, len(tvt_filled) - 1))
            tvt_matches.append(float(tvt_filled[center]))
            score_matches.append(float(score))

        # Pad to top_k if fewer matches
        while len(tvt_matches) < top_k:
            tvt_matches.append(tvt_matches[-1] if tvt_matches else 0.0)
            score_matches.append(0.0)

        tvt_arr = np.array(tvt_matches)
        score_arr = np.array(score_matches)

        # Weighted average (best TVT)
        weights = np.clip(score_arr, 0, None) + 1e-9
        best_tvt = float(np.average(tvt_arr, weights=weights))

        results.append({
            "best_tvt":      tvt_arr[0],
            "best_score":    score_arr[0],
            "second_tvt":    tvt_arr[1] if len(tvt_arr) > 1 else tvt_arr[0],
            "weighted_tvt":  best_tvt,
            "tvt_spread":    float(np.std(tvt_arr)),
            "top_k_tvts":    tvt_matches,
            "top_k_scores":  score_matches,
            "confidence":    float(score_arr[0]) if score_arr[0] > 0 else 0.0,
        })

    return results


def add_typewell_window_features(
    lateral_df:   pd.DataFrame,
    typewell_df:  pd.DataFrame,
    gr_col:       str = "GR",
    tvt_col:      str = "TVT_input",
    window_size:  int = 30,
    top_k:        int = 5,
) -> pd.DataFrame:
    """
    Add top-K typewell window match features to each row of the lateral.

    Added columns:
      - tw_best_tvt:        typewell TVT at best-matching position
      - tw_best_score:      correlation score of best match
      - tw_second_tvt:      typewell TVT at 2nd-best match
      - tw_weighted_tvt:    weighted average across top-K matches
      - tw_tvt_spread:      spread of top-K TVT values (match confidence)
      - tw_confidence:      best match score
      - tw_tvt_vs_ffill:    tw_best_tvt minus forward-filled TVT
    """
    out = lateral_df.copy()

    if gr_col not in lateral_df.columns or gr_col not in typewell_df.columns:
        return out

    lat_gr  = lateral_df[gr_col].interpolate(limit=5).fillna(lateral_df[gr_col].median()).values
    tw_gr   = typewell_df[gr_col].interpolate(limit=5).fillna(typewell_df[gr_col].median()).values
    tw_tvt  = typewell_df[tvt_col].values if tvt_col in typewell_df.columns else np.arange(len(tw_gr), dtype=float)

    print(f"[typewell_window] Matching {len(lat_gr)} lateral points to typewell of length {len(tw_gr)}...")
    matches = sliding_window_match(lat_gr, tw_gr, tw_tvt, window_size=window_size, top_k=top_k)

    out["tw_best_tvt"]      = [m["best_tvt"]     for m in matches]
    out["tw_best_score"]    = [m["best_score"]    for m in matches]
    out["tw_second_tvt"]    = [m["second_tvt"]    for m in matches]
    out["tw_weighted_tvt"]  = [m["weighted_tvt"]  for m in matches]
    out["tw_tvt_spread"]    = [m["tvt_spread"]    for m in matches]
    out["tw_confidence"]    = [m["confidence"]    for m in matches]

    if tvt_col in lateral_df.columns:
        tvt_ffill = lateral_df[tvt_col].ffill()
        out["tw_tvt_vs_ffill"] = out["tw_best_tvt"] - tvt_ffill.fillna(0)

    return out


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    if hi == lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _window_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    da, db = a - a.mean(), b - b.mean()
    denom = (np.sqrt(np.sum(da**2)) * np.sqrt(np.sum(db**2))) + 1e-9
    return float(np.dot(da, db) / denom)


def _suppress_near_duplicates(
    scores: list[tuple],
    window: int,
    top_k:  int,
) -> list[tuple]:
    """Keep top-K matches that are at least window/2 apart."""
    selected = []
    for score, start, center in scores:
        too_close = any(abs(center - s[2]) < window // 2 for s in selected)
        if not too_close:
            selected.append((score, start, center))
        if len(selected) >= top_k:
            break
    return selected


def _empty_result(top_k: int) -> dict:
    return {
        "best_tvt":     0.0,
        "best_score":   0.0,
        "second_tvt":   0.0,
        "weighted_tvt": 0.0,
        "tvt_spread":   0.0,
        "top_k_tvts":   [0.0] * top_k,
        "top_k_scores": [0.0] * top_k,
        "confidence":   0.0,
    }
