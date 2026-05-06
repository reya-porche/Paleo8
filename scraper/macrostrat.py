"""
geo_tvt/scraper/macrostrat.py
Scrapes Macrostrat (https://macrostrat.org) for:
  - Stratigraphic columns
  - Lithology data
  - Formation names and ages
  - Unit thickness and contacts

All results are cached locally for offline use.
"""

import requests
import time
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MACROSTRAT_BASE, REQUEST_TIMEOUT, REQUEST_RETRIES
from cache.storage import cache_get, cache_set


def _get(endpoint: str, params: dict, source_tag: str) -> Optional[dict]:
    """GET with retry logic and cache layer."""
    cached = cache_get(source_tag, params)
    if cached is not None:
        return cached

    url = f"{MACROSTRAT_BASE}/{endpoint}"
    for attempt in range(REQUEST_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            cache_set(source_tag, params, data)
            return data
        except requests.RequestException as e:
            if attempt == REQUEST_RETRIES - 1:
                print(f"[macrostrat] Failed after {REQUEST_RETRIES} attempts: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


# ─── Public API ──────────────────────────────────────────────────────────────

def get_columns_near(lat: float, lon: float, radius_km: float = 200) -> Optional[list]:
    """
    Fetch stratigraphic columns within radius_km of a lat/lon.
    Returns list of column metadata dicts.
    """
    data = _get("columns", {
        "lat": lat, "lng": lon,
        "adjacentColumns": int(radius_km / 50),
        "response": "long",
    }, source_tag="macrostrat_columns")

    if not data or "success" not in data:
        return None

    return data["success"].get("data", [])


def get_units_for_column(column_id: int) -> Optional[list]:
    """
    Fetch all stratigraphic units for a given column.
    Returns list of units with lithology, age, thickness.
    """
    data = _get("units", {
        "col_id": column_id,
        "response": "long",
    }, source_tag="macrostrat_units")

    if not data or "success" not in data:
        return None

    return data["success"].get("data", [])


def get_lithologies() -> Optional[list]:
    """Fetch the full Macrostrat lithology vocabulary."""
    data = _get("lithologies", {"response": "long"}, source_tag="macrostrat_liths")
    if not data or "success" not in data:
        return None
    return data["success"].get("data", [])


def get_formations_near(lat: float, lon: float) -> Optional[list]:
    """
    Get formation names and stratigraphic position near a location.
    Useful for cross-referencing competition formation labels.
    """
    cols = get_columns_near(lat, lon)
    if not cols:
        return None

    formations = []
    for col in cols[:5]:  # limit to 5 closest columns
        col_id = col.get("col_id")
        if not col_id:
            continue
        units = get_units_for_column(col_id)
        if units:
            for u in units:
                formations.append({
                    "col_id":        col_id,
                    "col_name":      col.get("col_name", ""),
                    "unit_id":       u.get("unit_id"),
                    "unit_name":     u.get("unit_name", ""),
                    "strat_name":    u.get("strat_name", ""),
                    "lith":          u.get("lith", ""),
                    "lith_type":     u.get("lith_type", ""),
                    "lith_class":    u.get("lith_class", ""),
                    "age_top":       u.get("t_age"),
                    "age_bottom":    u.get("b_age"),
                    "thickness_m":   u.get("max_thick"),
                    "environ":       u.get("environ", ""),
                    "pbdb_collections": u.get("pbdb_collections", 0),
                })
    return formations


def get_regional_lith_distribution(lat: float, lon: float) -> dict:
    """
    Summarize lithology class distribution across nearby columns.
    Returns {lith_class: fractional_proportion}.
    Used as a geological prior feature.
    """
    formations = get_formations_near(lat, lon) or []
    counts: dict = {}
    total = 0
    for f in formations:
        lc = f.get("lith_class", "unknown") or "unknown"
        counts[lc] = counts.get(lc, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def get_age_range_near(lat: float, lon: float) -> dict:
    """
    Return min/max geological age (Ma) of formations near a location.
    Useful for constraining paleogeographic reconstruction age.
    """
    formations = get_formations_near(lat, lon) or []
    ages = []
    for f in formations:
        if f.get("age_top") is not None:
            ages.append(f["age_top"])
        if f.get("age_bottom") is not None:
            ages.append(f["age_bottom"])
    if not ages:
        return {"age_min_ma": None, "age_max_ma": None}
    return {"age_min_ma": min(ages), "age_max_ma": max(ages)}
