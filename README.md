# GeoTVT — Geological TVT Prediction Pipeline

A hierarchical geological prior system for TVT prediction in horizontal drilling.
Combines paleogeographic history, live geological data scraping, and AI-powered
ontology mapping to improve sequence-based TVT prediction.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Layer 1 — Global Priors               │
│   EarthByte paleogeography · USGS tectonic provinces    │
│   Macrostrat lithology · Basin type · Climate proxy     │
└────────────────────┬────────────────────────────────────┘
                     │ geological prior vector
┌────────────────────▼────────────────────────────────────┐
│                    Layer 2 — Typewell Encoder            │
│   1D CNN + self-attention over typewell GR log          │
│   Output: geological context embedding [64-dim]         │
└────────────────────┬────────────────────────────────────┘
                     │ typewell embedding
┌────────────────────▼────────────────────────────────────┐
│                    Layer 3 — TVT Predictor               │
│   CatBoost baseline (fast, strong)                      │
│   GeoTVTTransformer (autoregressive, uncertainty-aware) │
│   Fuses: telemetry + typewell context + priors          │
└────────────────────┬────────────────────────────────────┘
                     │ TVT prediction + uncertainty
┌────────────────────▼────────────────────────────────────┐
│                    Layer 4 — Anomaly Detection           │
│   Stratigraphic continuity prior                        │
│   Claude AI geological interpretation                   │
│   Flags: fault crossing · formation boundary · mismatch │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Scrape geological context for your drill location
```bash
python main.py scrape --lat 31.5 --lon -102.0 --age 100 --output data/priors.json
```

### 4. Train model on competition data
```bash
python main.py train --data data/train.csv --prior-json data/priors.json
```

### 5. Predict on test data
```bash
python main.py predict --data data/test.csv --prior-json data/priors.json
```

### 6. Full pipeline (one command)
```bash
python main.py run --lat 31.5 --lon -102.0 --data data/train.csv
```

---

## Individual Tools

### Geological Ontology Mapper
Standardizes messy geological terminology using Claude AI:
```bash
python main.py ontology --desc "fine-grained siliciclastic unit with plant fossils"
```

### Anomaly Detection
Flags sensor readings that conflict with geological priors:
```bash
python main.py anomaly --gr 145 --tvt-delta 3.2 --prior-json data/priors.json
```

### Cache Management
All scraped data is cached locally for offline use:
```bash
python main.py cache --action stats
python main.py cache --action clear
```

---

## Data Sources

| Source | Data | API |
|--------|------|-----|
| **Macrostrat** | Lithology, formations, stratigraphy | https://macrostrat.org/api/v2 |
| **EarthByte GPlates** | Paleogeographic reconstruction | https://gws.gplates.org |
| **USGS MRDS/SGMC** | Mineral occurrences, geologic units | https://mrdata.usgs.gov |
| **Anthropic Claude** | Ontology mapping, anomaly interpretation | API key required |

All external data is cached in `cache/geo_cache.db` (SQLite).
System works fully offline after first scrape.

---

## Competition Strategy

**Phase 1 — Baseline** (submit fast)
- CatBoost with sequence features
- Geological priors as additional features
- Expected: measurable lift over sequence-only baseline

**Phase 2 — Transformer** (final submission)  
- GeoTVTTransformer with typewell encoder
- Autoregressive prediction in NaN zone
- Uncertainty quantification

**The core claim:**
> Historical Earth context (paleogeography, basin type, depositional environment)
> provides statistically significant additional signal for local TVT prediction
> beyond what drilling telemetry alone contains.

---

## File Structure

```
geo_tvt/
├── main.py                    # CLI entry point
├── config.py                  # Settings, API endpoints
├── requirements.txt
├── scraper/
│   ├── __init__.py            # Unified scrape_location()
│   ├── macrostrat.py          # Lithology, stratigraphy data
│   ├── earthbyte.py           # Paleogeographic reconstruction
│   └── usgs.py                # Geologic units, minerals
├── priors/
│   ├── __init__.py
│   ├── geological.py          # Feature engineering, continuity priors
│   └── ontology.py            # Claude AI term standardization
├── model/
│   ├── __init__.py
│   ├── typewell_encoder.py    # 1D CNN + attention encoder
│   ├── predictor.py           # CatBoost + Transformer models
│   └── trainer.py             # Training, evaluation, prediction
├── cache/
│   └── storage.py             # SQLite cache (offline mode)
└── data/                      # Put competition data here
```

---

## Notes

- The system gracefully degrades: if external APIs are down, it uses cached data.
  If cache is empty and APIs are down, it uses feature defaults with low confidence.
- Claude API is optional but strongly recommended for ontology mapping.
  Without it, a keyword-based fallback is used.
- For the competition, scrape priors for each unique well location in the dataset.
  Run `scrape_location()` in a loop and store to a JSON mapping by well_id.
