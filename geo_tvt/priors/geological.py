"""
geo_tvt/priors/geological.py
Converts scraped geological data into ML-ready prior features.

This is Layer 1 in the hierarchical system:
  Global history → Regional priors → Local prediction

Takes raw scraper output and produces:
  - Normalized feature vectors
  - Depositional environment probability distributions
  - Stratigraphic continuity priors
  - Physical plausibility constraints for TVT jumps
"""

import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEPOSITIONAL_ENVIRONMENTS, BASIN_TYPES


# Encoding maps for categorical features
CLIMATE_ZONE_ENC = {
    "tropical": 0, "subtropical": 1, "warm_temperate": 2,
    "cool_temperate": 3, "subpolar": 4, "polar": 5, "unknown": -1,
}
OCEAN_PROX_ENC = {
    "tropical_shelf": 0, "passive_margin_distal": 1,
    "epicontinental_sea": 2, "open_ocean_margin": 3,
    "continental_interior": 4, "modern_analog": 5, "uncertain": -1, "unknown": -1,
}
TECTONIC_ENC = {k: i for i, k in enumerate(BASIN_TYPES + ["unclassified", "unknown"])}
ENV_ENC      = {k: i for i, k in enumerate(DEPOSITIONAL_ENVIRONMENTS + ["unknown"])}


def build_prior_features(scraped: dict) -> dict:
    """
    Convert raw scrape_location() output into normalized numerical features.
    
    All features are scaled to roughly [-1, 1] or [0, 1] ranges.
    NaN values are filled with physically meaningful defaults.
    """
    f = {}

    # ── Paleogeographic features ────────────────────────────────────────────
    paleo_lat = scraped.get("paleo_lat") or scraped.get("lat", 0.0)
    paleo_lon = scraped.get("paleo_lon") or scraped.get("lon", 0.0)

    f["paleo_lat_norm"]     = paleo_lat / 90.0          # [-1, 1]
    f["paleo_lon_norm"]     = paleo_lon / 180.0         # [-1, 1]
    f["paleo_abs_lat_norm"] = abs(paleo_lat) / 90.0     # [0, 1]
    f["paleo_temp_norm"]    = (scraped.get("paleo_temp_proxy_c", 15.0) - 0) / 35.0

    f["climate_zone_enc"]   = CLIMATE_ZONE_ENC.get(
        scraped.get("paleo_climate_zone", "unknown"), -1
    ) / 5.0  # normalize to [-0.2, 1]

    f["ocean_prox_enc"]     = OCEAN_PROX_ENC.get(
        scraped.get("paleo_ocean_proximity", "unknown"), -1
    ) / 5.0

    # ── Lithology distribution ───────────────────────────────────────────────
    f["frac_sedimentary"]   = scraped.get("lith_sedimentary", 0.0)
    f["frac_igneous"]       = scraped.get("lith_igneous", 0.0)
    f["frac_metamorphic"]   = scraped.get("lith_metamorphic", 0.0)

    # ── Age features ────────────────────────────────────────────────────────
    age_min = scraped.get("regional_age_min_ma") or scraped.get("age_ma", 100.0)
    age_max = scraped.get("regional_age_max_ma") or scraped.get("age_ma", 100.0)

    f["age_min_norm"]    = np.log1p(age_min) / np.log1p(540)   # [0, 1], 540 = Cambrian
    f["age_max_norm"]    = np.log1p(age_max) / np.log1p(540)
    f["age_span_norm"]   = np.log1p(max(0, age_max - age_min)) / np.log1p(540)

    # ── Tectonic province ────────────────────────────────────────────────────
    tectonic_raw = scraped.get("tectonic_province", "unknown")
    f["tectonic_enc"] = TECTONIC_ENC.get(tectonic_raw, len(TECTONIC_ENC) - 1) / max(len(TECTONIC_ENC), 1)

    # ── Mineral activity ─────────────────────────────────────────────────────
    f["mineral_site_density"] = min(scraped.get("mineral_site_count", 0) / 50.0, 1.0)

    return f


def compute_stratigraphic_continuity_prior(
    prev_tvt_values: list[float],
    geological_features: dict,
) -> dict:
    """
    Compute a prior distribution over TVT change at the next step.
    
    Based on:
      1. Recent TVT trend (from drilling data)
      2. Geological environment (some environments have higher layer continuity)
      3. Tectonic setting (faulted basins have higher variance)
    
    Returns:
      - expected_delta_tvt: expected change in TVT
      - delta_tvt_std: expected standard deviation of the change
      - max_plausible_jump: maximum physically plausible TVT jump
    """
    if not prev_tvt_values:
        return {
            "expected_delta_tvt":   0.0,
            "delta_tvt_std":        0.5,
            "max_plausible_jump":   2.0,
        }

    # Trend from recent data
    recent = np.array(prev_tvt_values[-10:])
    if len(recent) > 1:
        deltas = np.diff(recent)
        expected_delta = float(np.mean(deltas))
        trend_std      = float(np.std(deltas))
    else:
        expected_delta = 0.0
        trend_std      = 0.5

    # Geological continuity modifier
    # Passive margins and cratonic basins: smoother
    # Rift basins and foreland: more variable
    tectonic_enc = geological_features.get("tectonic_enc", 0.5)
    continuity_factor = 1.0 + tectonic_enc  # 1.0 - 2.0

    # Sedimentary fraction: higher sed → more layer-continuous
    sed_frac = geological_features.get("frac_sedimentary", 0.5)
    continuity_factor *= (2.0 - sed_frac)  # more igneous → less continuous

    prior_std = trend_std * continuity_factor
    max_jump  = prior_std * 3.0  # 3-sigma plausibility bound

    return {
        "expected_delta_tvt":  expected_delta,
        "delta_tvt_std":       float(np.clip(prior_std, 0.1, 5.0)),
        "max_plausible_jump":  float(np.clip(max_jump, 0.3, 10.0)),
    }


def flag_geological_anomaly(
    gr_value: float,
    tvt_delta: float,
    geological_features: dict,
    continuity_prior: dict,
) -> dict:
    """
    Flag when observed sensor data conflicts with geological priors.
    
    This is the "anomaly detection" layer — the model's early warning system.
    When telemetry conflicts with prior expectations, it often signals:
      - Fault crossing
      - Formation boundary
      - Unexpected lithology
    """
    max_jump    = continuity_prior.get("max_plausible_jump", 2.0)
    tvt_anomaly = abs(tvt_delta) > max_jump

    # GR anomaly: very high GR often signals shale entry
    gr_high = gr_value > 100.0   # API units
    gr_low  = gr_value < 20.0    # clean carbonate / clean sand

    sed_frac = geological_features.get("frac_sedimentary", 0.5)

    # High GR in igneous-dominated region: unexpected
    gr_anomaly = gr_high and sed_frac < 0.3

    flags = []
    if tvt_anomaly:
        flags.append("unexpected_tvt_jump")
    if gr_anomaly:
        flags.append("gr_lithology_mismatch")
    if gr_high and tvt_anomaly:
        flags.append("possible_fault_crossing")

    return {
        "has_anomaly":   len(flags) > 0,
        "anomaly_flags": flags,
        "tvt_delta_zscore": abs(tvt_delta) / max(continuity_prior.get("delta_tvt_std", 0.5), 0.01),
    }


def feature_vector_from_scrape(scraped: dict, prev_tvt: list = None) -> np.ndarray:
    """
    Full feature vector as numpy array, ready for model input.
    Useful for batch processing.
    """
    base_features = build_prior_features(scraped)

    if prev_tvt:
        continuity = compute_stratigraphic_continuity_prior(prev_tvt, base_features)
        base_features["expected_delta_tvt"] = continuity["expected_delta_tvt"]
        base_features["delta_tvt_std"]      = continuity["delta_tvt_std"]
        base_features["max_plausible_jump"] = continuity["max_plausible_jump"]
    else:
        base_features["expected_delta_tvt"] = 0.0
        base_features["delta_tvt_std"]      = 0.5
        base_features["max_plausible_jump"] = 2.0

    return np.array(list(base_features.values()), dtype=np.float32)
