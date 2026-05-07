"""
geo_tvt/features/sedimentary_fingerprint.py
Sedimentary fingerprint features — the competition-viable form of the Pangea idea.

Instead of:
  "where were rocks during Pangea?"

We ask:
  "what depositional dynamics does this GR sequence encode?"

Ancient water systems (rivers, deltas, shallow seas) create RECURRING
sedimentary patterns. Those patterns appear as GR texture signatures.
Wells drilled through similar environments will have similar signatures,
regardless of their anonymized X/Y coordinates.

Features encoded:
  - Cyclicity score (rhythmic = marine/deltaic, aperiodic = continental)
  - Coarsening vs fining up trend (progradational vs retrogradational)
  - Shale-to-sand ratio (marine vs continental proxy)
  - Boundary sharpness (abrupt = channel/unconformity, gradational = marine)
  - High/low frequency spectral energy (depositional rhythm)
  - Upward-cleaning trend (fining-up = transgressive, coarsening-up = regressive)
"""

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks
from scipy.stats  import linregress
from typing import Optional


def build_sedimentary_fingerprint(
    df: pd.DataFrame,
    gr_col:   str = "GR",
    md_col:   str = "MD",
    tvt_col:  str = "TVT_input",
    window:   int = 100,    # rows to analyze for each fingerprint window
    step:     int = 10,     # stride between windows
) -> pd.DataFrame:
    """
    Compute sedimentary fingerprint features for each row.

    Each row gets:
      - Local depositional environment scores
      - GR spectral features (cyclicity, rhythm)
      - Trend direction (coarsening/fining up)
      - Boundary sharpness
      - Channel/marine/shale classification scores
    """
    feat = df.copy()
    n = len(feat)

    if gr_col not in feat.columns:
        return feat

    gr = feat[gr_col].interpolate(limit=5).fillna(feat[gr_col].median()).values
    md = feat[md_col].values if md_col in feat.columns else np.arange(n, dtype=float)

    # Output arrays (compute once per window, assign to all rows in window)
    cols = {
        "depo_cyclicity":       np.zeros(n),
        "depo_coarsening_up":   np.zeros(n),
        "depo_fining_up":       np.zeros(n),
        "depo_shale_fraction":  np.zeros(n),
        "depo_sand_fraction":   np.zeros(n),
        "depo_boundary_sharp":  np.zeros(n),
        "depo_channel_score":   np.zeros(n),
        "depo_marine_score":    np.zeros(n),
        "depo_spectral_low":    np.zeros(n),
        "depo_spectral_high":   np.zeros(n),
        "depo_gr_trend":        np.zeros(n),
        "depo_gr_var":          np.zeros(n),
        "depo_environment_enc": np.zeros(n),
    }

    # Compute fingerprints using sliding windows
    for start in range(0, n, step):
        end = min(start + window, n)
        segment = gr[start:end]
        if len(segment) < 10:
            continue

        fp = _compute_fingerprint(segment)

        for col_name, val in fp.items():
            if col_name in cols:
                cols[col_name][start:end] = val

    for col_name, arr in cols.items():
        feat[col_name] = arr

    # Per-well global statistics (full-well fingerprint)
    feat["well_gr_mean"]  = gr.mean()
    feat["well_gr_std"]   = gr.std()
    feat["well_gr_p10"]   = np.percentile(gr, 10)
    feat["well_gr_p90"]   = np.percentile(gr, 90)
    feat["well_gr_range"] = np.percentile(gr, 90) - np.percentile(gr, 10)

    # TVT fingerprint (how variable is TVT relative to GR changes)
    if tvt_col in feat.columns:
        tvt_known = feat[tvt_col].dropna()
        if len(tvt_known) > 10:
            feat["well_tvt_range"] = float(tvt_known.max() - tvt_known.min())
            feat["well_tvt_std"]   = float(tvt_known.std())
        else:
            feat["well_tvt_range"] = 0.0
            feat["well_tvt_std"]   = 0.0

    return feat.fillna(0)


def _compute_fingerprint(segment: np.ndarray) -> dict:
    """Compute all fingerprint scores for a GR segment."""
    n = len(segment)
    seg = segment - segment.mean()  # detrend

    # ── Cyclicity (dominant frequency) ───────────────────────────────────────
    try:
        freqs, psd = welch(segment, nperseg=min(n, 32))
        total_power = psd.sum() + 1e-9
        low_freq_power  = psd[:len(psd)//3].sum() / total_power
        high_freq_power = psd[len(psd)//3:].sum() / total_power
        # Cyclicity = concentration of power in mid-frequency (rhythmic deposition)
        mid_freq_power = psd[len(psd)//6:len(psd)//3].sum() / total_power
        cyclicity = float(mid_freq_power)
    except Exception:
        low_freq_power = high_freq_power = cyclicity = 0.0

    # ── Trend direction (coarsening vs fining up) ─────────────────────────────
    x = np.arange(n, dtype=float)
    try:
        slope, _, r, _, _ = linregress(x, segment)
    except Exception:
        slope, r = 0.0, 0.0

    # Negative slope = GR decreasing = coarsening up (more sand)
    # Positive slope = GR increasing = fining up (more shale)
    coarsening_up = float(max(0.0, -slope / (np.std(segment) + 1e-9)))
    fining_up     = float(max(0.0,  slope / (np.std(segment) + 1e-9)))
    gr_trend      = float(slope)

    # ── Compositional fractions ───────────────────────────────────────────────
    shale_frac = float(np.mean(segment > 100))
    sand_frac  = float(np.mean(segment < 60))

    # ── Boundary sharpness ────────────────────────────────────────────────────
    # Sharp boundaries = large abrupt GR changes (channel sands, unconformities)
    diffs = np.abs(np.diff(segment))
    p75 = np.percentile(diffs, 75)
    sharp_frac = float(np.mean(diffs > p75 * 2))

    # ── Channel score ─────────────────────────────────────────────────────────
    # Channel sands: sharp base, fining up, low GR
    channel_score = float(
        0.4 * sharp_frac +
        0.3 * fining_up +
        0.3 * sand_frac
    )

    # ── Marine shale score ────────────────────────────────────────────────────
    # Marine shale: high GR, low cyclicity, gradual
    marine_score = float(
        0.5 * shale_frac +
        0.3 * (1 - sharp_frac) +
        0.2 * (1 - cyclicity)
    )

    # ── Dominant environment encoding ─────────────────────────────────────────
    # Simple heuristic classification
    if channel_score > 0.5:
        env_enc = 1.0   # fluvial/channel
    elif marine_score > 0.5:
        env_enc = 2.0   # marine shale
    elif cyclicity > 0.3:
        env_enc = 3.0   # rhythmic (deltaic/tidal)
    elif sand_frac > 0.5:
        env_enc = 4.0   # clean sand (aeolian/beach)
    else:
        env_enc = 0.0   # mixed/unclassified

    return {
        "depo_cyclicity":       cyclicity,
        "depo_coarsening_up":   min(coarsening_up, 3.0),
        "depo_fining_up":       min(fining_up, 3.0),
        "depo_shale_fraction":  shale_frac,
        "depo_sand_fraction":   sand_frac,
        "depo_boundary_sharp":  sharp_frac,
        "depo_channel_score":   channel_score,
        "depo_marine_score":    marine_score,
        "depo_spectral_low":    float(low_freq_power),
        "depo_spectral_high":   float(high_freq_power),
        "depo_gr_trend":        float(np.clip(gr_trend, -5, 5)),
        "depo_gr_var":          float(np.std(segment)),
        "depo_environment_enc": env_enc,
    }


def build_sedimentary_features_for_dataset(
    df: pd.DataFrame,
    well_col: str = "well_id",
    **kwargs,
) -> pd.DataFrame:
    """Apply per-well sedimentary fingerprinting to the full dataset."""
    if well_col not in df.columns:
        return build_sedimentary_fingerprint(df, **kwargs)
    parts = []
    for _, grp in df.groupby(well_col, sort=False):
        parts.append(build_sedimentary_fingerprint(grp.copy(), **kwargs))
    return pd.concat(parts, ignore_index=True)
