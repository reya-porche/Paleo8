"""
geo_tvt/pipeline_competition.py
The main competition execution pipeline.

Execution order:
  Phase 1 — Sequence features + TVT continuation → submit baseline
  Phase 2 — Physics features + window typewell matching → measure lift
  Phase 3 — Sedimentary fingerprinting + clustering → measure lift
  Phase 4 — Hybrid physics predictor → final model

Every feature group is ablation-gated:
  Only added to the final model if it improves validation MAE.

Usage:
  python pipeline_competition.py --train data/train.csv --test data/test.csv
  python pipeline_competition.py --train data/train.csv --test data/test.csv --phase 2
"""

import sys
import click
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table

# Make sure sibling packages are importable when script is run directly
sys.path.insert(0, str(Path(__file__).parent))

console = Console()


def _load_data(path: str, well_col: str = "well_id") -> pd.DataFrame:
    """Load a CSV or a directory of CSVs, returning one concatenated DataFrame.

    When reading a directory each file is assumed to be one well.
    Prefers *__horizontal_well.csv files (Rogii competition naming) so that
    typewell CSV files in the same folder are never accidentally loaded.
    Falls back to all *.csv files only when no horizontal-well files are found.
    If the resulting DataFrame has no well_id column the filename stem is
    injected as well_id so the rest of the pipeline always has a well identifier.
    """
    p = Path(path)
    if p.is_dir():
        # Prefer competition-standard horizontal well naming
        csv_files = sorted(p.glob("*__horizontal_well.csv"))
        if not csv_files:
            csv_files = sorted(p.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        console.print(f"  Reading {len(csv_files)} CSV files from {path}")
        parts = []
        for f in csv_files:
            df = pd.read_csv(f)
            # Inject well identity from filename when the column is absent
            if well_col not in df.columns:
                df[well_col] = f.stem
            parts.append(df)
        return pd.concat(parts, ignore_index=True)
    if p.exists():
        return pd.read_csv(p)
    # Try appending .csv extension (common Kaggle path omission)
    p_csv = Path(str(path) + ".csv")
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Cannot find data at: {path}")


@click.command()
@click.option("--train",      required=True,  type=str)
@click.option("--test",       default=None,   type=str)
@click.option("--typewell",   default=None,   type=str)
@click.option("--target",     default="TVT",  type=str)
@click.option("--tvt-input",  default="TVT_input", type=str)
@click.option("--gr",         default="GR",   type=str)
@click.option("--md",         default="MD",   type=str)
@click.option("--x",          default="X",    type=str)
@click.option("--y",          default="Y",    type=str)
@click.option("--z",          default="Z",    type=str)
@click.option("--well-col",   default="well_id", type=str)
@click.option("--phase",      default=4,      type=int, help="Run phases 1 through N")
@click.option("--n-clusters", default=5,      type=int)
@click.option("--val-frac",   default=0.2,    type=float)
@click.option("--output",     default="submission.csv", type=str)
def run(train, test, typewell, target, tvt_input, gr, md, x, y, z,
        well_col, phase, n_clusters, val_frac, output):
    """
    Competition TVT prediction pipeline with ablation-gated feature layers.
    """
    console.print(Panel(
        f"[bold]GeoTVT Competition Pipeline[/]  Phase 1–{phase}\n"
        f"Train: {train}  |  Target: {target}",
        title="Paleo8 / GeoTVT"
    ))

    # ─── Load data ────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Loading data...[/]")
    train_df = _load_data(train, well_col=well_col)
    test_df  = _load_data(test,  well_col=well_col) if test else None

    n_wells = train_df[well_col].nunique() if well_col in train_df.columns else "?"
    console.print(f"  Train: {len(train_df):,} rows, {n_wells} wells")

    # Detect typewell
    if typewell:
        tw_df = pd.read_csv(typewell)
    elif "is_typewell" in train_df.columns:
        tw_df = train_df[train_df["is_typewell"] == 1].copy()
        train_df = train_df[train_df["is_typewell"] != 1].copy()
    else:
        tw_id = train_df.groupby(well_col)[gr].count().idxmax() if well_col in train_df.columns else None
        tw_df = train_df[train_df[well_col] == tw_id].copy() if tw_id else train_df.head(500).copy()
        console.print(f"  Auto-selected typewell: {tw_id}")

    # ─── Well split ───────────────────────────────────────────────────────────
    from model.trainer import split_wells
    train_split, val_split = split_wells(train_df, val_frac=val_frac)
    console.print(f"  Split: {train_split[well_col].nunique()} train wells, "
                  f"{val_split[well_col].nunique()} val wells")

    # Track running MAE as features are added
    active_features: list[str] = []
    best_val_mae = float("inf")

    # ─── Phase 1: Sequence + TVT continuation features ────────────────────────
    console.print("\n[bold cyan]Phase 1: Sequence + TVT continuation features...[/]")
    from features.sequence_features import build_features_for_dataset, get_feature_groups

    # train_feat uses only the train split so _quick_eval validation is uncontaminated
    train_feat = build_features_for_dataset(train_split, well_col=well_col,
                  gr_col=gr, md_col=md, x_col=x, y_col=y, z_col=z, tvt_col=tvt_input)
    val_feat   = build_features_for_dataset(val_split, well_col=well_col,
                  gr_col=gr, md_col=md, x_col=x, y_col=y, z_col=z, tvt_col=tvt_input)
    # full_feat is built from ALL train data for the final _finalize model
    full_feat  = build_features_for_dataset(train_df, well_col=well_col,
                  gr_col=gr, md_col=md, x_col=x, y_col=y, z_col=z, tvt_col=tvt_input)
    if test_df is not None:
        test_feat = build_features_for_dataset(test_df, well_col=well_col,
                     gr_col=gr, md_col=md, x_col=x, y_col=y, z_col=z, tvt_col=tvt_input)

    groups = get_feature_groups(train_feat)
    phase1_cols = (
        groups["baseline"] + groups["gr_rolling"] + groups["gr_derivatives"] +
        groups["gr_lags"] + groups["gr_proxies"] + groups["trajectory"] +
        groups["tvt_continuation"] + groups["tvt_history"]
    )
    phase1_cols = list(dict.fromkeys(c for c in phase1_cols if c in train_feat.columns))
    active_features = phase1_cols.copy()

    val_mae_p1 = _quick_eval(train_feat, val_feat, active_features, target, well_col)
    console.print(f"  Phase 1 Val MAE: [cyan]{val_mae_p1:.4f}[/]")
    best_val_mae = val_mae_p1

    if phase < 2:
        _finalize(full_feat, test_feat if test_df is not None else None,
                  active_features, target, well_col, output)
        return

    # ─── Phase 2: Physics features + window typewell ──────────────────────────
    console.print("\n[bold cyan]Phase 2: Physics features + window typewell matching...[/]")
    from features.physics_features import build_physics_features_for_dataset
    from alignment.window_typewell  import add_typewell_window_features

    def _apply_p2(df):
        df = build_physics_features_for_dataset(df, well_col=well_col,
              gr_col=gr, md_col=md, x_col=x, y_col=y, z_col=z, tvt_col=tvt_input)
        parts = []
        for _, grp in df.groupby(well_col):
            parts.append(add_typewell_window_features(grp, tw_df, gr_col=gr, tvt_col=tvt_input))
        return pd.concat(parts, ignore_index=True) if parts else df

    train_feat = _apply_p2(train_feat)
    val_feat   = _apply_p2(val_feat)
    full_feat  = _apply_p2(full_feat)
    if test_df is not None:
        test_feat = _apply_p2(test_feat)

    physics_cols = [c for c in train_feat.columns
                    if c not in active_features and c.startswith(
                        ("dz_", "dgr_", "d2", "dtvt", "curvature", "dogleg",
                         "formation_dip", "gr_z_coup", "geo_state", "transition",
                         "gr_savgol", "z_savgol", "path_speed", "azimuth")
                    )]
    tw_cols = [c for c in train_feat.columns if c.startswith("tw_")]
    candidate_p2 = physics_cols + tw_cols

    candidate_val_mae = _quick_eval(train_feat, val_feat, active_features + candidate_p2, target, well_col)
    if candidate_val_mae < best_val_mae - 0.001:
        console.print(f"  Phase 2 improves: {best_val_mae:.4f} -> [green]{candidate_val_mae:.4f}[/] OK")
        active_features += candidate_p2
        best_val_mae = candidate_val_mae
    else:
        console.print(f"  Phase 2 did NOT improve ({candidate_val_mae:.4f} vs {best_val_mae:.4f}), [yellow]skipping[/]")

    if phase < 3:
        _finalize(full_feat, test_feat if test_df is not None else None,
                  active_features, target, well_col, output)
        return

    # ─── Phase 3: Sedimentary fingerprint + clustering ────────────────────────
    console.print("\n[bold cyan]Phase 3: Sedimentary fingerprints + clustering...[/]")
    from features.sedimentary_fingerprint import build_sedimentary_features_for_dataset
    from clustering.well_clustering import build_well_summary_features, WellClusterer

    train_feat = build_sedimentary_features_for_dataset(train_feat, well_col=well_col,
                  gr_col=gr, md_col=md, tvt_col=tvt_input)
    val_feat   = build_sedimentary_features_for_dataset(val_feat, well_col=well_col,
                  gr_col=gr, md_col=md, tvt_col=tvt_input)
    full_feat  = build_sedimentary_features_for_dataset(full_feat, well_col=well_col,
                  gr_col=gr, md_col=md, tvt_col=tvt_input)
    if test_df is not None:
        test_feat = build_sedimentary_features_for_dataset(test_feat, well_col=well_col,
                     gr_col=gr, md_col=md, tvt_col=tvt_input)

    sed_cols = [c for c in train_feat.columns if c.startswith("depo_") or c.startswith("well_")]

    # Cluster wells — fit on full training set, apply to all splits
    summary_tr   = build_well_summary_features(full_feat, gr_col=gr, md_col=md,
                    x_col=x, y_col=y, z_col=z, tvt_col=tvt_input, well_col=well_col)
    # Clamp n_clusters so KMeans never requests more clusters than wells
    safe_k = max(2, min(n_clusters, len(summary_tr)))
    clusterer = WellClusterer(n_clusters=safe_k)
    clusterer.fit(summary_tr)
    train_feat = clusterer.assign_to_df(train_feat, summary_tr, well_col=well_col)
    val_feat   = clusterer.assign_to_df(val_feat,   summary_tr, well_col=well_col)
    full_feat  = clusterer.assign_to_df(full_feat,  summary_tr, well_col=well_col)
    if test_df is not None:
        summary_test = build_well_summary_features(test_feat, gr_col=gr, md_col=md,
                        x_col=x, y_col=y, z_col=z, tvt_col=tvt_input, well_col=well_col)
        test_feat = clusterer.assign_to_df(test_feat, summary_test, well_col=well_col)

    cluster_cols = ["cluster_id", "cluster_dist"]
    candidate_p3 = sed_cols + cluster_cols

    candidate_val_mae = _quick_eval(train_feat, val_feat, active_features + candidate_p3, target, well_col)
    if candidate_val_mae < best_val_mae - 0.001:
        console.print(f"  Phase 3 improves: {best_val_mae:.4f} -> [green]{candidate_val_mae:.4f}[/] OK")
        active_features += candidate_p3
        best_val_mae = candidate_val_mae
    else:
        console.print(f"  Phase 3 did NOT improve, [yellow]skipping[/]")

    if phase < 4:
        _finalize(full_feat, test_feat if test_df is not None else None,
                  active_features, target, well_col, output)
        return

    # ─── Phase 4: Hybrid physics predictor ────────────────────────────────────
    console.print("\n[bold cyan]Phase 4: Hybrid physics predictor...[/]")
    from physics.hybrid_predictor import HybridTVTPredictor

    hybrid = HybridTVTPredictor()
    # Fit on the full training data (not just the train split)
    hybrid.fit(full_feat[full_feat[target].notna()].copy(),
               well_col=well_col, tvt_col=target, gr_col=gr, z_col=z)

    train_feat = hybrid.predict(train_feat, well_col=well_col, tvt_col=tvt_input, gr_col=gr, z_col=z)
    val_feat   = hybrid.predict(val_feat,   well_col=well_col, tvt_col=tvt_input, gr_col=gr, z_col=z)
    full_feat  = hybrid.predict(full_feat,  well_col=well_col, tvt_col=tvt_input, gr_col=gr, z_col=z)
    if test_df is not None:
        test_feat = hybrid.predict(test_feat, well_col=well_col, tvt_col=tvt_input, gr_col=gr, z_col=z)

    hybrid_cols = ["tvt_physics_pred", "TVT_pred_physics"]
    candidate_p4 = hybrid_cols

    candidate_val_mae = _quick_eval(train_feat, val_feat, active_features + candidate_p4, target, well_col)
    if candidate_val_mae < best_val_mae - 0.001:
        console.print(f"  Phase 4 improves: {best_val_mae:.4f} -> [green]{candidate_val_mae:.4f}[/] OK")
        active_features += candidate_p4
        best_val_mae = candidate_val_mae
    else:
        console.print(f"  Phase 4 did NOT improve, [yellow]skipping[/]")

    _finalize(full_feat, test_feat if test_df is not None else None,
              active_features, target, well_col, output)

    console.print(Panel(
        f"[bold green]Pipeline complete![/]\n"
        f"Best Val MAE:     [cyan]{best_val_mae:.4f}[/]\n"
        f"Active features:  [cyan]{len(active_features)}[/]\n"
        f"Output:           {output}",
        title="Results"
    ))


# ─── Utilities ────────────────────────────────────────────────────────────────

def _quick_eval(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    feature_cols: list[str],
    target_col:   str,
    well_col:     str,
) -> float:
    """Quick CatBoost/GBM evaluation, returns val MAE."""
    avail = [c for c in feature_cols if c in train_df.columns and c in val_df.columns]
    if not avail:
        return float("inf")

    mask_tr = train_df[target_col].notna()
    mask_vl = val_df[target_col].notna()

    X_tr = train_df[mask_tr][avail].select_dtypes(include=[np.number]).fillna(0)
    y_tr = train_df[mask_tr][target_col].values
    X_vl = val_df[mask_vl][avail].select_dtypes(include=[np.number]).fillna(0).reindex(columns=X_tr.columns, fill_value=0)
    y_vl = val_df[mask_vl][target_col].values

    try:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=7,
                                  verbose=0, random_seed=42, loss_function="MAE")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=200, max_depth=7, random_state=42)

    model.fit(X_tr, y_tr)
    preds = model.predict(X_vl)
    return float(np.mean(np.abs(y_vl - preds)))


def _finalize(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame | None,
    feature_cols: list[str],
    target_col:   str,
    well_col:     str,
    output_path:  str,
) -> None:
    """Train final model on all training data and generate submission."""
    console.print("\n[bold cyan]Training final model on all data...[/]")
    avail = [c for c in feature_cols if c in train_df.columns]
    mask  = train_df[target_col].notna()
    X_all = train_df[mask][avail].select_dtypes(include=[np.number]).fillna(0)
    y_all = train_df[mask][target_col].values

    try:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=8,
                                  verbose=200, random_seed=42, loss_function="MAE")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=500, max_depth=8, random_state=42)

    model.fit(X_all, y_all)
    console.print(f"  [green]Final model trained on {len(y_all):,} samples, {len(avail)} features[/]")

    if test_df is not None:
        avail_test = [c for c in avail if c in test_df.columns]
        X_test = test_df[avail_test].select_dtypes(include=[np.number]).fillna(0).reindex(columns=X_all.columns, fill_value=0)
        preds = model.predict(X_test)

        submission = test_df[[well_col, "MD"]].copy() if "MD" in test_df.columns else test_df[[well_col]].copy()
        submission["TVT_predicted"] = preds
        submission.to_csv(output_path, index=False)
        console.print(f"  [green]Submission saved -> {output_path}[/]")


if __name__ == "__main__":
    run()
