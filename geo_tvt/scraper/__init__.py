"""
geo_tvt/scraper/__init__.py
Unified scraper interface. Call scrape_location() to collect
all geological context for a lat/lon point.
"""

from .macrostrat import get_formations_near, get_regional_lith_distribution, get_age_range_near
from .earthbyte  import build_paleo_feature_vector
from .usgs       import get_full_location_context


def scrape_location(lat: float, lon: float, age_ma: float = 100.0) -> dict:
    """
    Full geological context fetch for a location.
    Pulls from Macrostrat, EarthByte, and USGS in parallel-ish fashion.
    age_ma: representative formation age to use for paleogeographic reconstruction.

    Returns a flat dict ready for feature engineering.
    """
    print(f"  [scraper] Macrostrat...")
    lith_dist  = get_regional_lith_distribution(lat, lon)
    age_range  = get_age_range_near(lat, lon)

    print(f"  [scraper] EarthByte paleogeography...")
    paleo_vec  = build_paleo_feature_vector(lat, lon, age_ma)

    print(f"  [scraper] USGS location context...")
    usgs_ctx   = get_full_location_context(lat, lon)

    return {
        "lat": lat,
        "lon": lon,
        "age_ma": age_ma,

        # Macrostrat
        "lith_sedimentary":      lith_dist.get("sedimentary", 0.0),
        "lith_igneous":          lith_dist.get("igneous", 0.0),
        "lith_metamorphic":      lith_dist.get("metamorphic", 0.0),
        "lith_unknown":          lith_dist.get("unknown", 0.0),
        "regional_age_min_ma":   age_range.get("age_min_ma"),
        "regional_age_max_ma":   age_range.get("age_max_ma"),

        # EarthByte
        "paleo_lat":             paleo_vec.get("paleo_lat"),
        "paleo_lon":             paleo_vec.get("paleo_lon"),
        "paleo_climate_zone":    paleo_vec.get("paleo_climate_zone", "unknown"),
        "paleo_temp_proxy_c":    paleo_vec.get("paleo_temp_proxy_c", 15.0),
        "paleo_abs_latitude":    paleo_vec.get("paleo_abs_latitude"),
        "paleo_ocean_proximity": paleo_vec.get("paleo_ocean_proximity", "unknown"),

        # USGS
        "tectonic_province":     usgs_ctx.get("tectonic_province", "unknown"),
        "mineral_site_count":    usgs_ctx.get("mineral_site_count", 0),
        "primary_unit_age":      usgs_ctx.get("primary_unit_age", ""),
    }
