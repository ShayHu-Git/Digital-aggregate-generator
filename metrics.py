"""
metrics.py
==========
Numbers used to judge whether the idea works and is stable:

geometry fidelity   : RA / NA volume vs. the reference; adhered-mortar
                      content AMC = 1 - V_NA / V_RA vs. the reference
topology            : effective NA-core count vs. the reference
containment (the point of the thickness-field representation):
    - contain_sdf   : fraction of NA vertices where the generated d_RA > tol
                      -> should be ~0 with NO clipping, purely structural
    - contain_encl  : same thing measured independently with
                      select_enclosed_points (the test the rest of the
                      project already uses); small residuals here are the
                      known coincident-surface ray-cast noise, not real
                      penetration
diversity           : mean pairwise IoU of the { d_RA < 0 } masks across
                      seeds -- want clearly < 1 (not identical) and clearly
                      > 0 (not unrelated noise)
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates


def mesh_volume(mesh) -> float:
    if mesh is None or mesh.n_points == 0:
        return 0.0
    return abs(float(mesh.triangulate().volume))


def pct_delta(value: float, ref: float) -> float:
    return 100.0 * (value - ref) / ref if ref else float("nan")


def containment_sdf(mesh_na, d_ra_grid, origin, spacing, tol: float = 1e-4) -> float:
    """Fraction of NA-mesh vertices that fall outside the RA shell, measured
    by sampling the generated d_RA field itself (trilinear).  Structural
    expectation: exactly 0, since d_NA_recon = d_RA + T + margin >= d_RA."""
    if mesh_na is None or mesh_na.n_points == 0:
        return 0.0
    gc = (mesh_na.points - origin[None, :]) / spacing[None, :]
    vals = map_coordinates(d_ra_grid, gc.T, order=1, mode="nearest")
    return float(np.mean(vals > tol))


def containment_enclosed(mesh_na, mesh_ra) -> float:
    """Independent check with select_enclosed_points (ray casting)."""
    if mesh_na is None or mesh_na.n_points == 0 or mesh_ra is None:
        return 0.0
    sel = mesh_na.select_enclosed_points(mesh_ra.triangulate(), tolerance=1e-6,
                                         check_surface=False)
    inside = np.asarray(sel["SelectedPoints"], dtype=bool)
    return float(np.mean(~inside))


def exposed_fraction(mesh_na, t_grid, origin, spacing, tol: float) -> float:
    """Share of NA-surface vertices where the adhered-mortar thickness T is
    ~0 -- i.e. natural aggregate exposed at the particle surface (the
    partial-wrapping case).  Compare to the reference's own value."""
    if mesh_na is None or mesh_na.n_points == 0:
        return 0.0
    gc = (mesh_na.points - origin[None, :]) / spacing[None, :]
    t_at = map_coordinates(np.clip(t_grid, 0.0, None), gc.T, order=1, mode="nearest")
    return float(np.mean(t_at <= tol))


def pairwise_iou(grids) -> float:
    """grids: list of (n,n,n,2).  Mean pairwise IoU of the RA occupancy
    ({d_RA < 0}).  1.0 = identical shells, 0.0 = disjoint."""
    masks = [g[..., 0] < 0 for g in grids]
    if len(masks) < 2:
        return float("nan")
    ious = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum() or 1
            ious.append(inter / union)
    return float(np.mean(ious))
