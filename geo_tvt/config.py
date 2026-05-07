"""
geo_tvt/config.py
Central configuration for API keys, endpoints, and system settings.
Set ANTHROPIC_API_KEY in your environment before running.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
CACHE_DB    = BASE_DIR / "cache" / "geo_cache.db"
MODEL_DIR   = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
(BASE_DIR / "cache").mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ─── API Keys ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ─── External API Endpoints ───────────────────────────────────────────────────
MACROSTRAT_BASE   = "https://macrostrat.org/api/v2"
USGS_BASE         = "https://mrdata.usgs.gov/api/v1"
GPLATES_BASE      = "https://gws.gplates.org"
NOAA_PALEO_BASE   = "https://www.ncei.noaa.gov/access/paleo-search/api"

# ─── Scraper Settings ─────────────────────────────────────────────────────────
REQUEST_TIMEOUT   = 30          # seconds
REQUEST_RETRIES   = 3
CACHE_TTL_DAYS    = 30          # re-fetch after this many days

# ─── Model Settings ───────────────────────────────────────────────────────────
SEQUENCE_WINDOW   = 50          # MD steps to look back
PREDICTION_STEPS  = 10          # how many steps ahead to predict
CATBOOST_PARAMS = {
    "iterations":       1000,
    "learning_rate":    0.05,
    "depth":            8,
    "loss_function":    "RMSE",
    "eval_metric":      "RMSE",
    "random_seed":      42,
    "verbose":          100,
}

# ─── Geological Ontology ──────────────────────────────────────────────────────
# Known depositional environments and their paleo-proxies
DEPOSITIONAL_ENVIRONMENTS = [
    "fluvial_channel",
    "deltaic_plain",
    "shallow_marine",
    "deep_marine",
    "carbonate_platform",
    "evaporitic_basin",
    "eolian",
    "glacial",
    "volcanic_arc",
    "rift_basin",
    "foreland_basin",
    "passive_margin",
]

BASIN_TYPES = [
    "intracratonic",
    "rift",
    "passive_margin",
    "foreland",
    "strike_slip",
    "back_arc",
    "arc_related",
    "cratonic_sag",
]
