"""
geo_tvt/priors/ontology.py
Uses the Anthropic Claude API to standardize geological terminology.

The core problem: geological data uses inconsistent vocabulary.
  "fine sandstone" == "siliciclastic unit" == "deltaic sand body"
  "black shale" == "organic-rich mudrock" == "source rock facies"

This module maps any free-text geological description to a canonical
vocabulary, enabling consistent model features across all data sources.
"""

import json
import anthropic
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANTHROPIC_API_KEY, DEPOSITIONAL_ENVIRONMENTS
from cache.storage import cache_get, cache_set

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ─── Ontology Mapping ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a geological ontology expert. Your job is to standardize 
geological terminology into a canonical vocabulary. 

When given a geological description, you must return ONLY a JSON object with these fields:
{
  "canonical_lith_class": one of ["sedimentary", "igneous", "metamorphic", "unknown"],
  "canonical_lith_type": standardized rock type (e.g., "sandstone", "limestone", "shale", "granite"),
  "depositional_env": most likely environment from this list: """ + str(DEPOSITIONAL_ENVIRONMENTS) + """,
  "grain_size": one of ["clay", "silt", "fine_sand", "medium_sand", "coarse_sand", "gravel", "unknown"],
  "organic_richness": one of ["low", "moderate", "high", "unknown"],
  "porosity_class": one of ["tight", "moderate", "good", "unknown"],
  "confidence": float 0-1 representing your confidence in this mapping
}

Return ONLY the JSON. No explanation, no markdown, no preamble."""


def map_lith_description(description: str) -> dict:
    """
    Map any free-text lithology description to canonical vocabulary.
    Uses Claude API. Results are cached.
    """
    if not description or not description.strip():
        return _default_mapping()

    cache_key = {"desc": description.lower().strip()}
    cached = cache_get("ontology_map", cache_key)
    if cached:
        return cached

    if not ANTHROPIC_API_KEY:
        print("[ontology] No ANTHROPIC_API_KEY set. Using rule-based fallback.")
        return _rule_based_fallback(description)

    try:
        client = _get_client()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Map this geological description: {description}"}],
        )
        raw = msg.content[0].text.strip()
        result = json.loads(raw)
        cache_set("ontology_map", cache_key, result)
        return result
    except Exception as e:
        print(f"[ontology] Claude API error: {e}. Using fallback.")
        return _rule_based_fallback(description)


def map_batch(descriptions: list[str]) -> list[dict]:
    """Map a list of descriptions. Returns list of canonical dicts."""
    return [map_lith_description(d) for d in descriptions]


def interpret_anomaly(
    sensor_context: dict,
    geological_prior: dict,
    paleo_context: dict,
) -> str:
    """
    Ask Claude to interpret a sensor anomaly in geological context.
    This is the "expert reasoning" layer — translates raw signals into
    geologically meaningful language.

    sensor_context: dict with GR, resistivity, TVT, MD values
    geological_prior: dict from geological prior engine
    paleo_context: dict from earthbyte scraper
    """
    if not ANTHROPIC_API_KEY:
        return "API key not set. Cannot generate anomaly interpretation."

    prompt = f"""You are a drilling geologist analyzing an anomaly during horizontal well drilling.

Current sensor readings:
{json.dumps(sensor_context, indent=2)}

Regional geological prior:
{json.dumps(geological_prior, indent=2)}

Paleogeographic context:
{json.dumps(paleo_context, indent=2)}

In 2-3 sentences, describe what this anomaly most likely indicates geologically, 
and what the drilling engineer should watch for next. Be specific and technical."""

    try:
        client = _get_client()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"Interpretation unavailable: {e}"


# ─── Fallbacks ───────────────────────────────────────────────────────────────

def _default_mapping() -> dict:
    return {
        "canonical_lith_class":  "unknown",
        "canonical_lith_type":   "unknown",
        "depositional_env":      "unknown",
        "grain_size":            "unknown",
        "organic_richness":      "unknown",
        "porosity_class":        "unknown",
        "confidence":            0.0,
    }


def _rule_based_fallback(description: str) -> dict:
    """Simple keyword-based mapping when Claude API is unavailable."""
    desc = description.lower()
    result = _default_mapping()
    result["confidence"] = 0.4

    # Lith class
    if any(k in desc for k in ["sand", "shale", "limestone", "carbonate", "silt", "mud", "clay"]):
        result["canonical_lith_class"] = "sedimentary"
    elif any(k in desc for k in ["granite", "basalt", "volcanic", "intrusive"]):
        result["canonical_lith_class"] = "igneous"
    elif any(k in desc for k in ["gneiss", "schist", "quartzite", "metamorphic"]):
        result["canonical_lith_class"] = "metamorphic"

    # Lith type
    for lith in ["sandstone", "shale", "limestone", "mudstone", "siltstone",
                 "carbonate", "dolomite", "granite", "basalt", "coal"]:
        if lith in desc:
            result["canonical_lith_type"] = lith
            break

    # Depositional env
    env_keywords = {
        "deltaic_plain":      ["delta", "deltaic"],
        "fluvial_channel":    ["river", "fluvial", "channel"],
        "shallow_marine":     ["marine", "shelf", "shallow"],
        "deep_marine":        ["deep", "turbidite", "abyssal"],
        "carbonate_platform": ["reef", "carbonate platform", "bank"],
        "evaporitic_basin":   ["evaporite", "salt", "gypsum", "anhydrite"],
        "eolian":             ["aeolian", "eolian", "dune", "desert"],
    }
    for env, keywords in env_keywords.items():
        if any(k in desc for k in keywords):
            result["depositional_env"] = env
            break

    return result
