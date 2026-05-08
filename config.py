"""
geo_tvt/config.py
Central configuration for system settings.

FOR KAGGLE TRAINING: No external API keys needed!
  - All training uses only the training CSV
  - External APIs (Macrostrat, USGS, GPlates, NOAA) are public and optional
  - ANTHROPIC_API_KEY is only needed for the --ontology CLI command

Training command:
  python main.py train --data train.csv --strategy clustered
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
# OPTIONAL: Only needed for --ontology CLI command (not for training)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ─── External API Endpoints (all public, no auth required) ──────────────────────
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
    "iterations":         2000,
    "learning_rate":      0.03,
    "depth":              9,
    "loss_function":      "MAE",
    "eval_metric":        "MAE",
    "random_seed":        42,
    "verbose":            100,
    "l2_leaf_reg":        5,
    "bagging_temperature":0.2,
    "random_strength":    1.0,
    "bootstrap_type":     "Bayesian",
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
