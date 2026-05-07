"""
geo_tvt/scraper/earthbyte.py
Scrapes EarthByte GPlates Web Service for paleogeographic reconstructions.
  https://gws.gplates.org

Given a modern lat/lon and a geological age (Ma), returns:
  - Paleo-latitude and paleo-longitude (where was this point on ancient Earth?)
  - Distance to ancient coastline / ocean proxy
  - Tectonic plate context

This is the "memory of Earth" layer — connecting modern drill location
to its ancient depositional environment.
"""

import requests
import time
import math
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GPLATES_BASE, REQUEST_TIMEOUT, REQUEST_RETRIES
from cache.storage import cache_get, cache_set


def _get(endpoint: str, params: dict) -> Optional[dict]:
    cached = cache_get("earthbyte_" + endpoint, params)
    if cached is not None:
        return cached

    url = f"{GPLATES_BASE}/{endpoint}"
    for attempt in range(REQUEST_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            cache_set("earthbyte_" + endpoint, params, data)
            return data
        except requests.RequestException as e:
            if attempt == REQUEST_RETRIES - 1:
                print(f"[earthbyte] Failed: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


# ─── Public API ──────────────────────────────────────────────────────────────

def reconstruct_location(lat: float, lon: float, age_ma: float) -> Optional[dict]:
    """
    Reconstruct the paleo-position of a modern lat/lon at a given age (Ma).
    Returns paleo_lat, paleo_lon in the ancient reference frame.

    Example: A point in the Permian Basin at 280 Ma was in a very different
    continental position — this tells the model its ancient environmental context.
    """
    data = _get("reconstruct/point", {
        "lat": lat, "lng": lon,
        "time": age_ma,
        "model": "MERDITH2021",  # most current reconstruction model
    })

    if not data:
        return None

    coords = data.get("coordinates")
    if not coords or len(coords) < 2:
        return None

    return {
        "paleo_lat":    coords[1],
        "paleo_lon":    coords[0],
        "age_ma":       age_ma,
        "model":        "MERDITH2021",
    }


def get_paleoclimate_proxy(paleo_lat: float, age_ma: float) -> dict:
    """
    Infer broad paleoclimate zone from paleo-latitude and age.
    Returns climate zone label and estimated mean annual temperature proxy.

    Based on published Phanerozoic climate reconstructions.
    No API needed — uses physical rules.
    """
    abs_lat = abs(paleo_lat)

    # Approximate climate zones by paleo-latitude
    if abs_lat < 10:
        zone, temp_proxy = "tropical",       28.0
    elif abs_lat < 25:
        zone, temp_proxy = "subtropical",    24.0
    elif abs_lat < 40:
        zone, temp_proxy = "warm_temperate", 18.0
    elif abs_lat < 55:
        zone, temp_proxy = "cool_temperate", 10.0
    elif abs_lat < 70:
        zone, temp_proxy = "subpolar",        2.0
    else:
        zone, temp_proxy = "polar",          -8.0

    # Adjust for deep time: Cretaceous was ~4C warmer globally
    if age_ma > 66:
        temp_proxy += 4.0
    elif age_ma > 252:
        temp_proxy += 2.0

    return {
        "paleo_climate_zone":    zone,
        "paleo_temp_proxy_c":    temp_proxy,
        "paleo_abs_latitude":    abs_lat,
    }


def estimate_paleo_ocean_proximity(paleo_lat: float, paleo_lon: float, age_ma: float) -> dict:
    """
    Estimate proximity to ancient ocean margins using simplified continental
    geometry rules. Full coastline vector data requires large GIS files,
    so this uses a statistical proxy based on paleo-latitude and published
    ocean margin reconstructions.

    Returns:
      - estimated ocean proximity category
      - probable depositional environment range
    """
    abs_lat = abs(paleo_lat)

    # Continental interior signal: low paleo-lat + certain longitude bands
    # This is a heuristic approximation — production version should use
    # actual EarthByte coastline shapefiles loaded locally
    if 250 <= age_ma <= 300:  # Permo-Carboniferous — Pangea assembly
        if abs_lat < 20 and -30 < paleo_lon < 30:
            ocean_prox = "continental_interior"
            likely_env = ["fluvial_channel", "eolian", "lacustrine"]
        else:
            ocean_prox = "passive_margin_distal"
            likely_env = ["shallow_marine", "deltaic_plain", "carbonate_platform"]
    elif 65 <= age_ma <= 145:  # Cretaceous — high sea levels
        ocean_prox = "epicontinental_sea" if abs_lat < 50 else "open_ocean_margin"
        likely_env = ["shallow_marine", "carbonate_platform", "deep_marine"]
    elif age_ma < 2.6:  # Quaternary
        ocean_prox = "modern_analog"
        likely_env = ["fluvial_channel", "deltaic_plain", "shallow_marine"]
    else:
        ocean_prox = "uncertain"
        likely_env = ["unknown"]

    return {
        "paleo_ocean_proximity":  ocean_prox,
        "likely_environments":    likely_env,
        "environment_confidence": 0.6 if ocean_prox != "uncertain" else 0.3,
    }


def build_paleo_feature_vector(lat: float, lon: float, age_ma: float) -> dict:
    """
    Full paleogeographic feature vector for a location + age.
    This is the primary input to the geological prior engine.

    Combines:
      - Reconstructed paleo position
      - Climate proxy
      - Ocean proximity
      - Derived depositional environment probabilities
    """
    features = {
        "lat": lat, "lon": lon, "age_ma": age_ma,
        "paleo_lat": None, "paleo_lon": None,
        "paleo_climate_zone": "unknown",
        "paleo_temp_proxy_c": 15.0,
        "paleo_abs_latitude": abs(lat),
        "paleo_ocean_proximity": "unknown",
        "likely_environments": [],
        "environment_confidence": 0.0,
    }

    recon = reconstruct_location(lat, lon, age_ma)
    if recon:
        features["paleo_lat"] = recon["paleo_lat"]
        features["paleo_lon"] = recon["paleo_lon"]

        climate = get_paleoclimate_proxy(recon["paleo_lat"], age_ma)
        features.update(climate)

        ocean = estimate_paleo_ocean_proximity(
            recon["paleo_lat"], recon["paleo_lon"], age_ma
        )
        features.update(ocean)

    return features
