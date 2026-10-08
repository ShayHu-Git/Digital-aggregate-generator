"""
postprocess.py
==============
Thin clean-up layer applied to generated meshes.  It is deliberately small,
because the thickness-field representation already gives containment for
free -- this file only removes marching-cubes speckle and does the same
Taubin smoothing the rest of the project uses.

`count_cores` reports how many distinct NA cores the generated field holds
(connected components of {d_NA<0} above an ABSOLUTE volume floor). The
exported NA file always merges every core into one .ply as the user asked;
this count is for the report only.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

import morphology


def filter_small_components(mesh: pv.PolyData, min_volume: float = 0.3):
    """Drop connected components whose physical volume is < `min_volume`
    (mortar specks / marching-cubes slivers).  ABSOLUTE floor, not a
    fraction of the total -- see represent.MIN_CORE_MM3.  Returns
    (merged_mesh, n_kept)."""
    if mesh is None or mesh.n_points == 0:
        return mesh, 0
    try:
        bodies = mesh.connectivity("all").split_bodies()
    except Exception:
        return mesh.triangulate().clean(), 1
    parts = []
    for b in bodies:
        surf = b.extract_surface().triangulate().clean()
        if surf.n_points == 0:
            continue
        parts.append((abs(float(surf.volume)), surf))
    if not parts:
        return mesh.triangulate().clean(), 1
    kept = [s for v, s in parts if v >= min_volume]
    if not kept:
        kept = [max(parts, key=lambda t: t[0])[1]]
    out = kept[0]
    for s in kept[1:]:
        out = out.merge(s)
    return out.triangulate().clean(), len(kept)


def match_ra_volume(mesh_ra, mesh_na, target_volume: float):
    """Isotropically scale BOTH meshes about the RA centroid so the shell
    hits `target_volume`.  A similarity transform about a shared centre
    keeps the NA cores nested inside the shell exactly and leaves the AMC
    (NA/RA volume ratio) unchanged -- it only removes the mild net volume
    drift the random warp introduces.  Returns (mesh_ra, mesh_na, scale)."""
    if mesh_ra is None or mesh_ra.n_points == 0:
        return mesh_ra, mesh_na, 1.0
    v = abs(float(mesh_ra.triangulate().volume))
    if v <= 0:
        return mesh_ra, mesh_na, 1.0
    s = (float(target_volume) / v) ** (1.0 / 3.0)
    c = np.asarray(mesh_ra.center, dtype=np.float64)
    ra = mesh_ra.copy()
    ra.points = (c + (np.asarray(ra.points, np.float64) - c) * s).astype(np.float32)
    na = mesh_na
    if mesh_na is not None and mesh_na.n_points:
        na = mesh_na.copy()
        na.points = (c + (np.asarray(na.points, np.float64) - c) * s).astype(np.float32)
    return ra, na, s


def rescale_to_shape(mesh_ra, mesh_na, target_ei: float, target_fi: float,
                     b_target: float):
    """Anisotropic rescale of the particle to a user-defined morphology,
    done in the RA shell's PCA-oriented bounding-box frame, with the
    EI = b/a, FI = c/b convention (both <= 1):

        b  (intermediate box dimension) -> b_target        (from size fraction)
        a  (longest)                    -> b_target / EI
        c  (shortest)                   -> b_target * FI

    The SAME affine map (centre, axes, per-axis scale) is applied to the RA
    shell and to every NA core, so the cores stay nested inside the shell
    exactly (an affine map preserves inside/outside) and only the box
    proportions change.  Returns (mesh_ra, mesh_na, dict(target/achieved))."""
    if mesh_ra is None or mesh_ra.n_points == 0:
        return mesh_ra, mesh_na, {}
    a0, b0, c0, center, R = morphology.obb(mesh_ra.points)
    a_t, c_t = b_target / target_ei, b_target * target_fi
    s = np.array([a_t / a0, b_target / b0, c_t / c0], dtype=np.float64)

    def _xf(m):
        if m is None or m.n_points == 0:
            return m
        p = np.asarray(m.points, np.float64)
        local = (p - center) @ R          # world -> box frame
        local *= s                        # stretch per principal axis
        out = local @ R.T + center        # box frame -> world
        mm = m.copy()
        mm.points = out.astype(np.float32)
        return mm

    ra, na = _xf(mesh_ra), _xf(mesh_na)
    d = morphology.describe(ra)
    return ra, na, dict(
        target_EI=target_ei, achieved_EI=d["EI"],
        target_FI=target_fi, achieved_FI=d["FI"],
        target_b=b_target, achieved_b=d["b"],
        scale=tuple(float(x) for x in s),
    )


def taubin_smooth(mesh: pv.PolyData, n_iter: int = 15, pass_band: float = 0.1):
    if mesh is None or mesh.n_points == 0:
        return mesh
    return mesh.smooth_taubin(n_iter=n_iter, pass_band=pass_band)


def count_cores(d_na_grid: np.ndarray, spacing: np.ndarray, min_mm3: float = 0.3):
    """Primary NA-core count: connected components of {d_NA < 0} with volume
    >= `min_mm3`.  The generated field already separates the cores at N=64
    once T is warped by the same field as d_RA (single-field warp); this
    just applies the absolute-volume floor."""
    from scipy.ndimage import label

    lab, n = label(d_na_grid < 0)
    if n == 0:
        return 0
    vols = np.bincount(lab.ravel())[1:] * float(np.prod(spacing))
    return int((vols >= min_mm3).sum())
