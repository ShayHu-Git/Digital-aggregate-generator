"""
reconstruct.py
==============
Turn a generated (n, n, n, 2) grid [d_RA, T] into two triangle meshes:

    RA shell : zero level set of  d_RA
    NA cores : zero level set of  d_NA = d_RA + clip(T, 0, inf) + margin

Containment  { d_NA < 0 } subset { d_RA < 0 }  holds because
d_NA >= d_RA pointwise (T >= 0, margin >= 0).  This is the whole reason for
the thickness-field representation and needs no clipping / repair.

Smoothing is done in FIELD space with one linear Gaussian applied to BOTH
d_RA and d_NA.  A positive linear filter preserves the inequality
(d_NA_s - d_RA_s = Gaussian(clip(T) + margin) >= 0), so the smoothed meshes
are still exactly nested -- unlike smoothing the two meshes separately,
which lets NA vertices drift back out through the shell.

`margin = 0` lets the two surfaces touch (exposed-NA / partial-wrapping
case); any `margin > 0` keeps a guaranteed mortar gap.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from skimage import measure


def _level_set_mesh(field, origin, spacing, level=0.0):
    lo, hi = float(field.min()), float(field.max())
    if not (lo < level < hi):
        return None
    # Pad one voxel of guaranteed-"outside" value around the field so the
    # isosurface always closes INSIDE the array even when the shape reaches
    # the grid boundary (e.g. a large warp, or k=0 shells at high alpha).
    # Without this, marching_cubes leaves the surface open along any face
    # the shape touches -> a non-watertight mesh.
    pad_val = hi + max(1.0, abs(hi))
    fp = np.pad(field, 1, mode="constant", constant_values=pad_val)
    verts, faces, _, _ = measure.marching_cubes(fp, level=level, spacing=tuple(spacing))
    verts = verts - np.asarray(spacing)[None, :] + origin[None, :]  # undo the 1-voxel pad
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    return pv.PolyData(verts, faces_pv)


def smoothed_fields(grid, margin=0.0, smooth_sigma=0.6):
    d_ra = grid[..., 0].astype(np.float32)
    d_na = d_ra + np.clip(grid[..., 1], 0.0, None) + float(margin)
    if smooth_sigma and smooth_sigma > 0:
        d_ra = gaussian_filter(d_ra, sigma=smooth_sigma)
        d_na = gaussian_filter(d_na, sigma=smooth_sigma)
    return d_ra, d_na


def grid_to_meshes(grid, origin, spacing, margin=0.0, smooth_sigma=0.6):
    """Returns (mesh_ra, mesh_na, d_ra_smoothed).  d_ra_smoothed is handed
    back so the containment metric can sample the *exact* field the NA mesh
    was cut against."""
    d_ra, d_na = smoothed_fields(grid, margin=margin, smooth_sigma=smooth_sigma)
    return (_level_set_mesh(d_ra, origin, spacing),
            _level_set_mesh(d_na, origin, spacing),
            d_ra)
