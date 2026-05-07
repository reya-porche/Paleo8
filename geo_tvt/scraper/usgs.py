"""
geo_tvt/scraper/usgs.py
Scrapes USGS public APIs for:
  - Mineral occurrence records
  - Geologic map units
  - Tectonic province data
  - Rock unit descriptions

Data Source: https://mrdata.usgs.gov
"""

import requests
import time
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import USGS_BASE, REQUEST_TIMEOUT, REQUEST_RETRIES
from cache.storage import cache_get, cache_set

# USGS SGMCv2 geologic map API (Geological Map of North America units)
SGMC_BASE = "https://mrdata.usgs.gov/geology/us/api"


def _get(base: str, endpoint: str, params: dict, tag: str) -> Optional[dict]:
    cached = cache_get(tag, params)
    if cached is not None:
        return cached

    url = f"{base}/{endpoint}"
    for attempt in range(REQUEST_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            cache_set(tag, params, data)
            return data
        except requests.RequestException as e:
            if attempt == REQUEST_RETRIES - 1:
                print(f"[usgs] Failed ({tag}): {e}")
                return None
            time.sleep(2 ** attempt)
    return None


# ─── Public API ──────────────────────────────────────────────────────────────

def get_geologic_units_at(lat: float, lon: float) -> Optional[list]:
    """
    Get USGS geologic map units at a lat/lon point.
    Returns rock unit names, ages, lithology descriptions.
    Uses the USGS SGMC (State Geologic Map Compilation) API.
    """
    data = _get(SGMC_BASE, "geounit", {
        "lat": lat, "lng": lon,
        "output": "json",
    }, tag="usgs_geounit")

    if not data:
        return None

    units = data.get("geounit", [])
    if isinstance(units, dict):
        units = [units]

    return [
        {
            "unit_name":    u.get("unitname", ""),
            "unit_age":     u.get("age", ""),
            "lithology":    u.get("lithology", ""),
            "description":  u.get("description", ""),
            "state":        u.get("state", ""),
            "source_map":   u.get("sourcemap", ""),
        }
        for u in units
    ]


def get_mineral_occurrences_near(lat: float, lon: float, radius_km: float = 100) -> Optional[list]:
    """
    Fetch mineral occurrence records near a location from USGS MRDS.
    Returns deposit names, commodities, rock types.
    Useful for inferring lithology type from nearby extraction history.
    """
    data = _get(USGS_BASE, "mrds", {
        "lat": lat, "lng": lon,
        "radius": radius_km,
        "format": "json",
        "maxrec": 50,
    }, tag="usgs_mrds")

    if not data:
        return None

    results = data.get("results", {}).get("result", [])
    if isinstance(results, dict):
        results = [results]

    return [
        {
            "deposit_name":   r.get("dep_name", ""),
            "commodities":    r.get("commod1", ""),
            "rock_type":      r.get("rrtype", ""),
            "dev_status":     r.get("dev_stat", ""),
            "lat":            r.get("latitude"),
            "lon":            r.get("longitude"),
        }
        for r in results
    ]


def get_tectonic_province(lat: float, lon: float) -> Optional[str]:
    """
    Infer tectonic province from USGS geologic unit data.
    Maps geological age + unit description to a tectonic category.
    """
    units = get_geologic_units_at(lat, lon)
    if not units:
        return None

    age_text = " ".join(u.get("unit_age", "") for u in units).lower()
    desc_text = " ".join(u.get("description", "") for u in units).lower()

    # Rule-based province classification from text signals
    if any(k in desc_text for k in ["rift", "graben", "half-graben"]):
        return "rift_basin"
    if any(k in desc_text for k in ["thrust", "fold belt", "foreland"]):
        return "foreland_basin"
    if any(k in desc_text for k in ["passive margin", "shelf", "continental margin"]):
        return "passive_margin"
    if any(k in desc_text for k in ["volcanic", "arc", "caldera"]):
        return "volcanic_arc"
    if any(k in desc_text for k in ["craton", "shield", "basement"]):
        return "intracratonic"
    if any(k in age_text for k in ["paleozoic", "precambrian"]):
        return "cratonic_sag"

    return "unclassified"


def get_full_location_context(lat: float, lon: float) -> dict:
    """
    Aggregate all USGS data for a location into one dict.
    Used as input to the geological prior engine.
    """
    geo_units = get_geologic_units_at(lat, lon) or []
    minerals  = get_mineral_occurrences_near(lat, lon) or []
    tectonic  = get_tectonic_province(lat, lon)

    # Summarize commodity types nearby
    commodity_types = list({
        m["commodities"] for m in minerals
        if m.get("commodities")
    })

    # Summarize rock types nearby
    rock_types = list({
        u["lithology"] for u in geo_units
        if u.get("lithology")
    })

    return {
        "tectonic_province":    tectonic or "unknown",
        "surface_rock_types":   rock_types,
        "nearby_commodities":   commodity_types,
        "geo_unit_count":       len(geo_units),
        "mineral_site_count":   len(minerals),
        "primary_unit_name":    geo_units[0]["unit_name"] if geo_units else "",
        "primary_unit_age":     geo_units[0]["unit_age"] if geo_units else "",
    }
