"""
geo_tvt/coord_probe.py
Detects whether X/Y/Z in the competition data are:
  A) Real-world coordinates (lat/lon or UTM) → enrich with scrapers
  B) Anonymized local field coordinates       → use internal methods only

Run this FIRST before deciding your enrichment strategy.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def probe_coordinates(df: pd.DataFrame,
                      x_col: str = "X",
                      y_col: str = "Y",
                      z_col: str = "Z") -> dict:
    """
    Analyze coordinate columns to determine their reference frame.

    Returns a dict with:
      - coord_type: "latlon" | "utm" | "local_feet" | "local_meters" | "unknown"
      - real_world: bool — whether scraping is viable
      - confidence: 0-1
      - notes: list of diagnostic strings
    """
    notes = []
    result = {
        "coord_type":  "unknown",
        "real_world":  False,
        "confidence":  0.0,
        "notes":       notes,
        "x_range":     None,
        "y_range":     None,
        "z_range":     None,
        "x_span":      None,
        "y_span":      None,
    }

    missing = [c for c in [x_col, y_col] if c not in df.columns]
    if missing:
        notes.append(f"Missing columns: {missing}")
        return result

    x = df[x_col].dropna()
    y = df[y_col].dropna()
    z = df[z_col].dropna() if z_col in df.columns else pd.Series([], dtype=float)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = x_max - x_min
    y_span = y_max - y_min

    result["x_range"] = (x_min, x_max)
    result["y_range"] = (y_min, y_max)
    result["x_span"]  = x_span
    result["y_span"]  = y_span

    if z.any():
        result["z_range"] = (float(z.min()), float(z.max()))

    # ── Test: lat/lon ────────────────────────────────────────────────────────
    lat_lon_x = -180 <= x_min and x_max <= 180
    lat_lon_y = -90  <= y_min and y_max <= 90
    if lat_lon_x and lat_lon_y and x_span < 30 and y_span < 30:
        result["coord_type"] = "latlon"
        result["real_world"] = True
        result["confidence"] = 0.95
        notes.append(f"X in [{x_min:.3f}, {x_max:.3f}]: lat/lon range ✓")
        notes.append("→ REAL WORLD: Scrape Macrostrat + EarthByte aggressively.")
        return result

    # ── Test: UTM easting/northing ───────────────────────────────────────────
    utm_x = 100_000 <= x_min and x_max <= 900_000
    utm_y = 0 <= y_min and y_max <= 10_000_000
    if utm_x and utm_y:
        result["coord_type"] = "utm"
        result["real_world"] = True
        result["confidence"] = 0.85
        notes.append(f"X in [{x_min:.0f}, {x_max:.0f}]: UTM easting range ✓")
        notes.append("→ REAL WORLD: Convert to lat/lon, then scrape.")
        notes.append("  Use: pyproj.Transformer.from_crs('EPSG:326XX', 'EPSG:4326')")
        return result

    # ── Test: local feet ─────────────────────────────────────────────────────
    if x_span < 100_000 and y_span < 100_000 and abs(x_min) < 100_000:
        result["coord_type"] = "local_feet"
        result["real_world"] = False
        result["confidence"] = 0.80
        notes.append(f"X span={x_span:.0f}, Y span={y_span:.0f}: local field coords (feet)")
        notes.append("→ ANONYMIZED: Shift to internal methods.")
        notes.append("  Strategy: typewell matching, GR alignment, formation geometry.")
        return result

    # ── Test: local meters ───────────────────────────────────────────────────
    if x_span < 30_000 and y_span < 30_000:
        result["coord_type"] = "local_meters"
        result["real_world"] = False
        result["confidence"] = 0.75
        notes.append(f"X span={x_span:.0f}m, Y span={y_span:.0f}m: local field coords (meters)")
        notes.append("→ ANONYMIZED: Shift to internal methods.")
        return result

    notes.append("Could not classify coordinates. Inspect manually.")
    return result


def print_probe_report(report: dict) -> None:
    from rich.console import Console
    from rich.panel   import Panel
    from rich         import print as rprint

    console = Console()
    color = "green" if report["real_world"] else "yellow"
    symbol = "✓" if report["real_world"] else "⚠"

    console.print(Panel(
        f"[bold]Coordinate Type:[/]  [{color}]{report['coord_type']}[/]  {symbol}\n"
        f"[bold]Real-world?[/]       [{color}]{report['real_world']}[/]\n"
        f"[bold]Confidence:[/]       {report['confidence']:.0%}\n"
        f"[bold]X range:[/]          {report['x_range']}\n"
        f"[bold]Y range:[/]          {report['y_range']}\n"
        f"[bold]Z range:[/]          {report['z_range']}\n\n"
        + "\n".join(f"  • {n}" for n in report["notes"]),
        title="[bold]Coordinate Probe[/]"
    ))

    if not report["real_world"]:
        console.print(
            "\n[bold yellow]Anonymized coordinates detected.[/]\n"
            "External scraping will not add signal.\n"
            "Activating internal geological representation pipeline:\n"
            "  → typewell_matcher.py    (GR sequence alignment)\n"
            "  → formation_geometry.py  (surface interpolation)\n"
            "  → tvt_continuation.py    (autoregressive prediction)\n"
            "  → cross_well_repr.py     (representation learning)\n"
        )
