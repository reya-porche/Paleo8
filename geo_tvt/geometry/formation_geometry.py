"""
geo_tvt/geometry/formation_geometry.py
Interpolates formation tops across multiple wells to build
a structural surface model of the field.

This answers:
  "Given the X/Y position of a well, what depth should
   each formation top be at?"

Why this matters for TVT:
  TVT measures position relative to stratigraphy.
  If you know where the formation surfaces are in 3D space,
  you can infer TVT without GPS — just from structural geometry.

Methods:
  1. Thin-plate spline interpolation (smooth, geological)
  2. IDW (Inverse Distance Weighting) (robust fallback)
  3. Structural dip estimation (are layers tilting?)
  4. Formation thickness map (does the layer thin/thicken?)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial     import cKDTree
from typing import Optional


# ─── Surface Interpolation ────────────────────────────────────────────────────

def fit_formation_surface(
    well_x: np.ndarray,
    well_y: np.ndarray,
    formation_depths: np.ndarray,
    method: str = "rbf",
    smoothing: float = 1.0,
) -> object:
    """
    Fit an interpolating surface through known formation top depths.

    Args:
      well_x, well_y:       X/Y positions of known wells [n_wells]
      formation_depths:     depth to formation top at each well [n_wells]
      method:               "rbf" (smooth) or "idw" (robust)
      smoothing:            RBF smoothing factor (higher = smoother surface)

    Returns: interpolator object. Call predict_formation_depth() to query.
    """
    valid = np.isfinite(formation_depths)
    x, y, z = well_x[valid], well_y[valid], formation_depths[valid]

    if len(x) < 3:
        # Not enough control points: return constant mean surface
        mean_depth = float(np.mean(z)) if len(z) > 0 else 0.0
        return _ConstantSurface(mean_depth)

    points = np.column_stack([x, y])

    if method == "rbf":
        try:
            return RBFInterpolator(points, z, smoothing=smoothing, kernel="thin_plate_spline")
        except Exception:
            pass  # fall through to IDW

    # IDW fallback
    return _IDWSurface(points, z)


def predict_formation_depth(
    interpolator,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> np.ndarray:
    """
    Predict formation top depth at query locations.
    """
    query = np.column_stack([query_x, query_y])
    if isinstance(interpolator, (_ConstantSurface, _IDWSurface)):
        return interpolator(query)
    return interpolator(query)


class _ConstantSurface:
    def __init__(self, value: float):
        self.value = value
    def __call__(self, points):
        return np.full(len(points), self.value)


class _IDWSurface:
    def __init__(self, points: np.ndarray, values: np.ndarray, power: float = 2.0):
        self.tree   = cKDTree(points)
        self.values = values
        self.power  = power

    def __call__(self, query: np.ndarray) -> np.ndarray:
        dists, idxs = self.tree.query(query, k=min(5, len(self.values)))
        dists = np.atleast_2d(dists)
        idxs  = np.atleast_2d(idxs)
        # Avoid division by zero for exact matches
        dists = np.where(dists < 1e-10, 1e-10, dists)
        weights = 1.0 / dists ** self.power
        weights /= weights.sum(axis=1, keepdims=True)
        return (weights * self.values[idxs]).sum(axis=1)


# ─── Structural Analysis ──────────────────────────────────────────────────────

def estimate_structural_dip(
    well_x: np.ndarray,
    well_y: np.ndarray,
    formation_depths: np.ndarray,
) -> dict:
    """
    Estimate the regional dip of a formation surface.

    Fits a plane: depth = a*x + b*y + c
    Returns dip magnitude (ft/ft or m/m) and dip direction (degrees).
    """
    valid = np.isfinite(formation_depths)
    x, y, z = well_x[valid], well_y[valid], formation_depths[valid]

    if len(x) < 3:
        return {"dip_magnitude": 0.0, "dip_azimuth_deg": 0.0, "dip_x": 0.0, "dip_y": 0.0}

    # Least-squares plane fit
    A = np.column_stack([x, y, np.ones_like(x)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        dip_x, dip_y, intercept = coeffs
    except np.linalg.LinAlgError:
        return {"dip_magnitude": 0.0, "dip_azimuth_deg": 0.0, "dip_x": 0.0, "dip_y": 0.0}

    dip_mag = float(np.sqrt(dip_x**2 + dip_y**2))
    dip_az  = float(np.degrees(np.arctan2(dip_x, dip_y)) % 360)

    return {
        "dip_magnitude":  dip_mag,       # ft/ft (depth change per horizontal unit)
        "dip_azimuth_deg": dip_az,       # direction of maximum dip
        "dip_x":          float(dip_x),  # dip component in X direction
        "dip_y":          float(dip_y),  # dip component in Y direction
        "intercept":      float(intercept),
    }


def compute_formation_thickness_map(
    well_x: np.ndarray,
    well_y: np.ndarray,
    top_depths: np.ndarray,
    base_depths: np.ndarray,
) -> object:
    """
    Build an interpolated thickness map for a formation interval.
    Useful for inferring whether the drill is approaching a thin zone.
    """
    valid = np.isfinite(top_depths) & np.isfinite(base_depths)
    thicknesses = base_depths[valid] - top_depths[valid]
    thicknesses = np.clip(thicknesses, 0, None)  # thickness ≥ 0
    return fit_formation_surface(well_x[valid], well_y[valid], thicknesses)


# ─── Feature Generation ───────────────────────────────────────────────────────

class FormationGeometryFeaturizer:
    """
    Pre-fits formation surfaces from all available wells,
    then generates geometry features for any new well location.

    Usage:
      featurizer = FormationGeometryFeaturizer()
      featurizer.fit(known_wells_df, formation_tops_df)
      features   = featurizer.transform(query_well_df)
    """

    def __init__(self):
        self.surfaces  = {}   # formation_name → interpolator
        self.dips      = {}   # formation_name → dip dict
        self.thicknesses = {} # formation_name → interpolator

    def fit(
        self,
        wells_df: pd.DataFrame,        # columns: well_id, X, Y
        tops_df:  pd.DataFrame,        # columns: well_id, formation, depth, [base_depth]
    ) -> "FormationGeometryFeaturizer":
        """Fit surface models from known formation top data."""
        merged = tops_df.merge(wells_df[["well_id", "X", "Y"]], on="well_id", how="left")

        for formation, grp in merged.groupby("formation"):
            x = grp["X"].values
            y = grp["Y"].values
            d = grp["depth"].values

            self.surfaces[formation] = fit_formation_surface(x, y, d)
            self.dips[formation]     = estimate_structural_dip(x, y, d)

            if "base_depth" in grp.columns:
                b = grp["base_depth"].values
                self.thicknesses[formation] = compute_formation_thickness_map(x, y, d, b)

        print(f"[geometry] Fit surfaces for {len(self.surfaces)} formations.")
        return self

    def transform(
        self,
        well_df: pd.DataFrame,
        x_col: str = "X",
        y_col: str = "Y",
        tvt_col: str = "TVT_input",
    ) -> pd.DataFrame:
        """
        Generate geometry features for each row of a well.

        Adds per-formation predicted depths and distances.
        Also adds structural dip-corrected TVT estimate.
        """
        out = well_df.copy()
        x = well_df[x_col].values
        y = well_df[y_col].values

        for formation, surface in self.surfaces.items():
            fname = formation.replace(" ", "_").lower()
            predicted_depth = predict_formation_depth(surface, x, y)
            out[f"pred_top_{fname}"] = predicted_depth

            # Distance from current TVT to predicted formation top
            if tvt_col in well_df.columns:
                tvt = well_df[tvt_col].fillna(well_df[tvt_col].median())
                out[f"dist_to_{fname}"] = tvt.values - predicted_depth

            # Thickness at this position
            if formation in self.thicknesses:
                thickness = predict_formation_depth(self.thicknesses[formation], x, y)
                out[f"thick_{fname}"] = thickness

            # Regional dip contribution
            dip = self.dips.get(formation, {})
            if dip.get("dip_magnitude", 0) > 0:
                dip_correction = dip["dip_x"] * x + dip["dip_y"] * y
                out[f"dip_corr_{fname}"] = dip_correction

        return out

    def get_structural_context(self, x: float, y: float) -> dict:
        """
        Return a summary of structural context at a single X/Y location.
        Used for anomaly interpretation.
        """
        context = {}
        for formation, surface in self.surfaces.items():
            fname = formation.replace(" ", "_").lower()
            depth = float(predict_formation_depth(surface, np.array([x]), np.array([y]))[0])
            context[f"pred_{fname}"] = depth
            dip = self.dips.get(formation, {})
            context[f"dip_{fname}"] = dip.get("dip_magnitude", 0.0)
        return context
