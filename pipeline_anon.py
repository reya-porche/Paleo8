"""
geo_tvt/pipeline_anon.py
Master pipeline for the anonymized-coordinate case.

Replaces external scraping with:
  1. Coordinate detection (verify anonymization)
  2. Typewell GR alignment (DTW)
  3. Formation surface geometry (structural model)
  4. Cross-well representation learning (geological embeddings)
  5. TVT continuation (Kalman + AR + similar-well transfer)
  6. Ablation study (measure what actually helps)

Run: python pipeline_anon.py --train data/train.csv --test data/test.csv
"""

import click
import json
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.panel   import Panel

console = Console()

# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--train",      required=True,  type=str, help="Training data CSV")
@click.option("--test",       default=None,   type=str, help="Test data CSV for predictions")
@click.option("--typewell",   default=None,   type=str, help="Typewell CSV (if separate file)")
@click.option("--target",     default="TVT",  type=str, help="Target column name")
@click.option("--gr-col",     default="GR",   type=str)
@click.option("--md-col",     default="MD",   type=str)
@click.option("--x-col",      default="X",    type=str)
@click.option("--y-col",      default="Y",    type=str)
@click.option("--well-col",   default="well_id", type=str)
@click.option("--ablate/--no-ablate", default=True, help="Run ablation study")
@click.option("--repr-epochs", default=30,    type=int, help="Encoder training epochs")
def run(train, test, typewell, target, gr_col, md_col, x_col, y_col, well_col,
        ablate, repr_epochs):
    """
    Full anonymized-coordinate TVT prediction pipeline.
    """
    console.print(Panel(
        "[bold]GeoTVT — Anonymized Coordinate Pipeline[/]\n"
        "Typewell alignment · Formation geometry · Cross-well representations",
        title="Pipeline Start"
    ))

    # ── Step 0: Load data ────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 0: Loading data...[/]")
    train_df = pd.read_csv(train)
    console.print(f"  Train: {len(train_df):,} rows, {train_df[well_col].nunique() if well_col in train_df.columns else '?'} wells")

    test_df = pd.read_csv(test) if test else None
    if test_df is not None:
        console.print(f"  Test:  {len(test_df):,} rows")

    # ── Step 1: Probe coordinates ────────────────────────────────────────────
    console.print("\n[bold cyan]Step 1: Probing coordinates...[/]")
    from coord_probe import probe_coordinates, print_probe_report
    probe = probe_coordinates(train_df, x_col, y_col)
    print_probe_report(probe)

    if probe["real_world"]:
        console.print("[green]Real-world coordinates detected! Consider running main.py for scraping.[/]")

    # ── Step 2: Typewell identification ──────────────────────────────────────
    console.print("\n[bold cyan]Step 2: Identifying typewell...[/]")

    if typewell:
        tw_df = pd.read_csv(typewell)
        console.print(f"  Loaded external typewell: {len(tw_df):,} rows")
    elif "is_typewell" in train_df.columns:
        tw_df = train_df[train_df["is_typewell"] == 1].copy()
        train_df = train_df[train_df["is_typewell"] != 1].copy()
        console.print(f"  Split typewell from train: {len(tw_df):,} typewell rows")
    else:
        # Use the well with the most GR coverage as typewell proxy
        if well_col in train_df.columns:
            tw_id = train_df.groupby(well_col)[gr_col].count().idxmax()
            tw_df = train_df[train_df[well_col] == tw_id].copy()
            console.print(f"  Auto-selected typewell: {tw_id} ({len(tw_df):,} rows)")
        else:
            tw_df = train_df.head(500).copy()
            console.print("  Using first 500 rows as typewell proxy")

    # ── Step 3: Typewell alignment ───────────────────────────────────────────
    console.print("\n[bold cyan]Step 3: Typewell GR alignment...[/]")
    from alignment.typewell_matcher import alignment_features

    train_aligned = alignment_features(train_df, tw_df, gr_col=gr_col, md_col=md_col)
    console.print(f"  Added alignment features: {[c for c in train_aligned.columns if c not in train_df.columns]}")

    if test_df is not None:
        test_aligned = alignment_features(test_df, tw_df, gr_col=gr_col, md_col=md_col)
    else:
        test_aligned = None

    # ── Step 4: Formation geometry ───────────────────────────────────────────
    console.print("\n[bold cyan]Step 4: Building formation surface geometry...[/]")

    if well_col in train_df.columns and x_col in train_df.columns:
        from geometry.formation_geometry import FormationGeometryFeaturizer

        # Build well-level summary for surface fitting
        wells_summary = train_df.groupby(well_col)[[x_col, y_col]].mean().reset_index()
        wells_summary.columns = ["well_id", "X", "Y"]

        # Formation tops: use min depth per well as proxy if no explicit tops
        if "formation" in train_df.columns and "formation_top" in train_df.columns:
            tops_df = train_df.groupby([well_col, "formation"])["formation_top"].first().reset_index()
            tops_df.columns = ["well_id", "formation", "depth"]
        else:
            # Create synthetic formation proxy from TVT quartile zones
            def tvt_zones(grp):
                tvt_known = grp[target if target in grp else "TVT_input"].dropna()
                if len(tvt_known) < 10:
                    return pd.DataFrame()
                q25, q75 = tvt_known.quantile(0.25), tvt_known.quantile(0.75)
                rows = []
                for name, depth in [("zone_low", q25), ("zone_high", q75)]:
                    rows.append({well_col: grp[well_col].iloc[0], "formation": name, "depth": depth})
                return pd.DataFrame(rows)

            if well_col in train_df.columns:
                tops_df = pd.concat([tvt_zones(g) for _, g in train_df.groupby(well_col)], ignore_index=True)
                tops_df = tops_df.rename(columns={well_col: "well_id"})
            else:
                tops_df = pd.DataFrame(columns=["well_id", "formation", "depth"])

        featurizer = FormationGeometryFeaturizer()
        if len(tops_df) > 0 and len(wells_summary) > 0:
            featurizer.fit(wells_summary, tops_df)
            train_aligned = featurizer.transform(train_aligned, x_col, y_col)
            if test_aligned is not None:
                test_aligned = featurizer.transform(test_aligned, x_col, y_col)
            console.print(f"  Formation geometry features added.")
        else:
            console.print("  [yellow]Skipping geometry: insufficient formation top data[/]")
    else:
        console.print("  [yellow]Skipping geometry: missing well_id or X/Y columns[/]")

    # ── Step 5: Cross-well representation learning ───────────────────────────
    console.print(f"\n[bold cyan]Step 5: Training GR encoder ({repr_epochs} epochs)...[/]")
    from representation.cross_well_repr import train_encoder, WellEmbeddingIndex

    encoder = train_encoder(train_aligned, n_epochs=repr_epochs)

    index = WellEmbeddingIndex()
    if well_col in train_aligned.columns:
        index.build(encoder, train_aligned, gr_col=gr_col, well_id_col=well_col)
        index.save()
        console.print("  [green]Well embedding index built.[/]")

    # ── Step 6: Feature engineering ──────────────────────────────────────────
    console.print("\n[bold cyan]Step 6: Full feature engineering...[/]")
    from model.predictor import engineer_sequence_features

    # Collect all geo feature column names
    geo_cols = [c for c in train_aligned.columns if c not in train_df.columns]

    train_featured = engineer_sequence_features(train_aligned)
    baseline_cols  = [c for c in train_featured.columns if c not in train_aligned.columns or c in train_df.columns]
    alignment_cols = [c for c in geo_cols if "aligned" in c or "gr_dev" in c or "formation_prox" in c]
    geometry_cols  = [c for c in geo_cols if "pred_top" in c or "dist_to" in c or "dip_corr" in c or "thick" in c]
    repr_cols      = []  # Would add embedding similarity features here

    console.print(f"  Total features: {len(train_featured.columns)}")
    console.print(f"  Baseline: {len(baseline_cols)}, Alignment: {len(alignment_cols)}, Geometry: {len(geometry_cols)}")

    # ── Step 7: Ablation study ───────────────────────────────────────────────
    if ablate:
        console.print("\n[bold cyan]Step 7: Ablation study...[/]")
        from model.trainer import split_wells
        from evaluation.ablation import run_standard_ablation

        val_frac = 0.2
        tr, vl = split_wells(train_featured, val_frac=val_frac)

        study = run_standard_ablation(tr, vl, baseline_cols, alignment_cols, geometry_cols, repr_cols, target_col=target)
        study.run()
        study.print_report()
        study.save_report()

    # ── Step 8: Train final model ────────────────────────────────────────────
    console.print("\n[bold cyan]Step 8: Training final model...[/]")
    from model.trainer import train_baseline
    metrics = train_baseline(train, geo_priors=None)
    console.print(f"  [green]Val MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}[/]")

    # ── Step 9: Generate test predictions ───────────────────────────────────
    if test_df is not None:
        console.print("\n[bold cyan]Step 9: Generating test predictions...[/]")
        from model.trainer import predict_test
        preds = predict_test(test, model_name="catboost_tvt")
        console.print(f"  [green]Predictions saved.[/]")

    console.print(Panel("[bold green]Pipeline complete![/]", title="Done"))


if __name__ == "__main__":
    run()
