"""
geo_tvt/main.py
CLI entry point for the GeoTVT pipeline.

Usage:
  python main.py scrape   --lat 31.5 --lon -102.0 --age 100
  python main.py train    --data data/train.csv
  python main.py predict  --data data/test.csv
  python main.py cache    --action stats
  python main.py ontology --desc "fine-grained siliciclastic unit with plant fossils"
  python main.py anomaly  --gr 145 --tvt-delta 3.2

Set environment variable: ANTHROPIC_API_KEY=your_key_here
"""

import os
import sys
import json
import click
from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich         import print as rprint
from pathlib      import Path

console = Console()

# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """
    ╔══════════════════════════════════════════╗
    ║  GeoTVT — Geological TVT Prediction     ║
    ║  Hierarchical Prior + Sequence Model    ║
    ╚══════════════════════════════════════════╝
    
    Combines paleogeographic history, Macrostrat lithology data,
    USGS geological context, and Claude AI ontology mapping to
    improve TVT prediction in horizontal drilling.
    """
    pass


# ─── scrape ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--lat",    required=True,  type=float, help="Latitude of drill location")
@click.option("--lon",    required=True,  type=float, help="Longitude of drill location")
@click.option("--age",    default=100.0,  type=float, help="Formation age in Ma (default: 100)")
@click.option("--output", default=None,   type=str,   help="Save JSON to file")
def scrape(lat, lon, age, output):
    """Scrape geological context for a lat/lon location."""
    from scraper import scrape_location

    console.print(Panel(
        f"[bold cyan]Scraping geological context[/]\n"
        f"Location: ({lat}, {lon})\n"
        f"Target age: {age} Ma",
        title="GeoTVT Scraper"
    ))

    with console.status("[bold green]Fetching from Macrostrat, EarthByte, USGS..."):
        result = scrape_location(lat, lon, age_ma=age)

    # Display results
    table = Table(title="Geological Prior Features", show_header=True)
    table.add_column("Feature", style="cyan")
    table.add_column("Value",   style="green")

    for k, v in result.items():
        if v is not None:
            table.add_row(k, str(round(v, 4) if isinstance(v, float) else v))

    console.print(table)

    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        console.print(f"[bold green]Saved to {output}[/]")


# ─── train ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--data",       required=True, type=str, help="Path to training CSV")
@click.option("--prior-json", default=None,  type=str, help="Path to geo prior JSON (from scrape)")
@click.option("--model",      default="catboost", type=click.Choice(["catboost", "transformer"]))
def train(data, prior_json, model):
    """Train TVT prediction model on competition data."""
    from model.trainer import train_baseline

    geo_priors = None
    if prior_json and os.path.exists(prior_json):
        with open(prior_json) as f:
            geo_priors = json.load(f)
        console.print(f"[cyan]Loaded geological priors from {prior_json}[/]")

    console.print(Panel(
        f"[bold]Training {model.upper()} model[/]\n"
        f"Data: {data}\n"
        f"Geological priors: {'YES' if geo_priors else 'NO (use --prior-json to add)'}",
        title="GeoTVT Trainer"
    ))

    if model == "catboost":
        metrics = train_baseline(data, geo_priors=geo_priors)
        console.print(f"\n[bold green]Training complete![/]")
        console.print(f"  Val MAE:  [cyan]{metrics['mae']:.4f}[/]")
        console.print(f"  Val RMSE: [cyan]{metrics['rmse']:.4f}[/]")
        console.print(f"  Features: [cyan]{metrics['n_features']}[/]")
    else:
        console.print("[yellow]Transformer training not yet wired to CLI — use model/trainer.py directly[/]")


# ─── predict ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--data",       required=True, type=str, help="Path to test CSV")
@click.option("--prior-json", default=None,  type=str, help="Path to geo prior JSON")
@click.option("--model-name", default="catboost_tvt", type=str)
def predict(data, prior_json, model_name):
    """Generate TVT predictions on test data."""
    from model.trainer import predict_test

    geo_priors = None
    if prior_json and os.path.exists(prior_json):
        with open(prior_json) as f:
            geo_priors = json.load(f)

    console.print(Panel(f"Predicting TVT on: {data}", title="GeoTVT Predictor"))
    result = predict_test(data, model_name=model_name, geo_priors=geo_priors)
    console.print(f"[green]Predicted {len(result)} rows.[/]")
    console.print(result.head(10).to_string())


# ─── ontology ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--desc", required=True, type=str, help="Geological description to standardize")
def ontology(desc):
    """Map a geological description to canonical vocabulary using Claude AI."""
    from priors.ontology import map_lith_description

    console.print(f"[cyan]Mapping:[/] {desc}")

    with console.status("[bold green]Querying Claude ontology mapper..."):
        result = map_lith_description(desc)

    table = Table(title="Canonical Mapping", show_header=True)
    table.add_column("Field",  style="cyan")
    table.add_column("Value",  style="green")
    table.add_column("Confidence")

    conf = result.pop("confidence", 0.0)
    for k, v in result.items():
        table.add_row(k, str(v), "")
    table.add_row("confidence", f"{conf:.2f}", "★" * int(conf * 5))

    console.print(table)


# ─── anomaly ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--gr",         required=True,  type=float, help="Gamma ray reading (API units)")
@click.option("--tvt-delta",  required=True,  type=float, help="TVT change at this step")
@click.option("--prior-json", default=None,   type=str,   help="Path to geo prior JSON")
def anomaly(gr, tvt_delta, prior_json):
    """Flag and interpret a sensor anomaly against geological priors."""
    from priors.geological import (
        compute_stratigraphic_continuity_prior,
        flag_geological_anomaly,
        build_prior_features,
    )
    from priors.ontology import interpret_anomaly

    geo_features = {}
    paleo_context = {}

    if prior_json and os.path.exists(prior_json):
        with open(prior_json) as f:
            scraped = json.load(f)
        geo_features  = build_prior_features(scraped)
        paleo_context = {k: scraped[k] for k in ["paleo_lat", "paleo_lon",
                          "paleo_climate_zone", "paleo_ocean_proximity"]
                         if k in scraped}

    # Simulate a recent TVT sequence
    fake_prev_tvt = list(range(10))  # placeholder — use real data in practice
    continuity = compute_stratigraphic_continuity_prior(fake_prev_tvt, geo_features)
    flags = flag_geological_anomaly(gr, tvt_delta, geo_features, continuity)

    # Display
    console.print(Panel(
        f"GR: [bold]{gr}[/] API\n"
        f"TVT Δ: [bold]{tvt_delta:+.3f}[/]\n"
        f"Max plausible Δ: [cyan]{continuity['max_plausible_jump']:.3f}[/]\n"
        f"Anomaly: [{'bold red' if flags['has_anomaly'] else 'green'}]{flags['has_anomaly']}[/]\n"
        f"Flags: {', '.join(flags['anomaly_flags']) or 'none'}\n"
        f"TVT Z-score: {flags['tvt_delta_zscore']:.2f}",
        title="Anomaly Detection"
    ))

    if flags["has_anomaly"] and os.getenv("ANTHROPIC_API_KEY"):
        console.print("[cyan]Querying Claude for geological interpretation...[/]")
        sensor_ctx = {"GR": gr, "TVT_delta": tvt_delta}
        interpretation = interpret_anomaly(sensor_ctx, geo_features, paleo_context)
        console.print(Panel(interpretation, title="[bold]Claude Geological Interpretation[/]"))


# ─── cache ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--action", type=click.Choice(["stats", "clear"]), default="stats")
@click.option("--source", default=None, type=str, help="Clear specific source only")
def cache(action, source):
    """Manage the local geological data cache."""
    from cache.storage import cache_stats, cache_clear

    if action == "stats":
        stats = cache_stats()
        if not stats:
            console.print("[yellow]Cache is empty.[/]")
        else:
            table = Table(title="Cache Statistics")
            table.add_column("Source",  style="cyan")
            table.add_column("Entries", style="green")
            table.add_column("Oldest")
            table.add_column("Newest")
            for src, s in stats.items():
                table.add_row(src, str(s["entries"]), s["oldest"] or "", s["newest"] or "")
            console.print(table)

    elif action == "clear":
        n = cache_clear(source=source)
        console.print(f"[green]Cleared {n} cache entries.[/]")


# ─── full pipeline ────────────────────────────────────────────────────────────

@cli.command("run")
@click.option("--lat",  required=True, type=float)
@click.option("--lon",  required=True, type=float)
@click.option("--age",  default=100.0, type=float)
@click.option("--data", required=True, type=str, help="Training CSV")
def run_pipeline(lat, lon, age, data):
    """
    Full pipeline: scrape → build priors → train → evaluate.
    
    This is the one-shot command to go from raw data to trained model.
    """
    import tempfile, os
    from scraper import scrape_location
    from model.trainer import train_baseline
    from priors.geological import build_prior_features

    console.print(Panel("[bold]GeoTVT Full Pipeline[/]", subtitle="scrape → features → train"))

    # Step 1: Scrape
    console.print("\n[bold cyan]Step 1: Scraping geological context...[/]")
    scraped = scrape_location(lat, lon, age_ma=age)

    # Step 2: Build priors
    console.print("\n[bold cyan]Step 2: Building geological prior features...[/]")
    prior_features = build_prior_features(scraped)
    console.print(f"  Generated {len(prior_features)} prior features")

    # Save priors to temp file
    prior_path = Path(data).parent / "geo_priors.json"
    with open(prior_path, "w") as f:
        json.dump(scraped, f, indent=2)
    console.print(f"  Saved priors → {prior_path}")

    # Step 3: Train
    console.print("\n[bold cyan]Step 3: Training model...[/]")
    metrics = train_baseline(data, geo_priors=prior_features)

    console.print(Panel(
        f"[bold green]Pipeline complete![/]\n\n"
        f"Val MAE:  [cyan]{metrics['mae']:.4f}[/]\n"
        f"Val RMSE: [cyan]{metrics['rmse']:.4f}[/]\n"
        f"Features: [cyan]{metrics['n_features']}[/] (incl. {len(prior_features)} geological priors)",
        title="Results"
    ))


if __name__ == "__main__":
    cli()
