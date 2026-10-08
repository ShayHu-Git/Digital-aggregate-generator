"""
viz.py
======
Two static figures for eyeballing the result (pure matplotlib / Agg):

  out/montage_slices.png : rows = references (A2, A7, A8), cols = reference +
      generated seeds.  Each panel is a mid cross-section of the [d_RA, T]
      grid -- grey = inside the RA shell, red = inside an NA core, dark line
      = RA outline.  This is the clearest view of "cores inside the shell,
      mortar rim between, layout varies per seed".

  out/montage_<ref>.png  : the same particles as 3D surface point clouds.

Grids for the seeds are re-generated from the reference with the same seed
(synth.generate is deterministic), so this script needs no extra files
from run.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import represent
import reconstruct
import synth

HERE = Path(__file__).resolve().parent
REF = represent.REF_DIR
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
NAMES = ("A2", "A7", "A8")
SEEDS = (1, 2, 3, 4)
ALPHA = 0.5


def _slice_panel(ax, d_ra, d_na, title):
    """Cross-section at the grid slice (axis 0) through the NA centroid."""
    inside_na = d_na < 0
    idx = int(np.round(np.argwhere(inside_na)[:, 0].mean())) if inside_na.any() else d_ra.shape[0] // 2
    ra_s, na_s = d_ra[idx], d_na[idx]
    ax.contourf(ra_s.T, levels=[-1e9, 0], colors=["0.78"])
    ax.contourf(na_s.T, levels=[-1e9, 0], colors=["#c0392b"])
    ax.contour(ra_s.T, levels=[0], colors=["0.25"], linewidths=0.8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def montage_slices():
    fig, axes = plt.subplots(len(NAMES), 1 + len(SEEDS),
                             figsize=(2.1 * (1 + len(SEEDS)), 2.1 * len(NAMES)))
    for r, name in enumerate(NAMES):
        grid, origin, spacing, meta = represent.load_reference(represent.build_reference(name, n=64))
        d_ra, d_na = reconstruct.smoothed_fields(grid, margin=0.0, smooth_sigma=0.6)
        _slice_panel(axes[r, 0], d_ra, d_na, f"{name} reference")
        for c, sd in enumerate(SEEDS, start=1):
            g = synth.generate(grid, seed=sd, alpha=ALPHA, verbose=False)
            gd_ra, gd_na = reconstruct.smoothed_fields(g, margin=0.0, smooth_sigma=0.6)
            _slice_panel(axes[r, c], gd_ra, gd_na, f"seed {sd}")
    fig.suptitle("Mid cross-section: grey = inside RA shell, red = inside NA core "
                 "(cores stay within the shell every seed)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "montage_slices.png", dpi=140); plt.close(fig)
    print(f"wrote {OUT / 'montage_slices.png'}")


def _pts(path, k=6000, seed=0):
    m = pv.read(str(path))
    p = np.asarray(m.points, np.float64)
    if len(p) > k:
        p = p[np.random.default_rng(seed).choice(len(p), k, replace=False)]
    return p


def _cloud_panel(ax, ra_pts, na_pts, title):
    c = ra_pts.mean(0)
    for pset, col, a, s in ((ra_pts, "0.55", 0.10, 4), (na_pts, "#c0392b", 0.55, 5)):
        if len(pset):
            ax.scatter(pset[:, 0], pset[:, 1], pset[:, 2], s=s, c=col, alpha=a,
                       edgecolors="none", depthshade=True)
    r = 0.72 * np.abs(ra_pts - c).max()
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=35)
    ax.set_title(title, fontsize=10)


def montage_cloud(name):
    cols = 1 + len(SEEDS)
    fig = plt.figure(figsize=(2.7 * cols, 3.1))
    ax = fig.add_subplot(1, cols, 1, projection="3d")
    _cloud_panel(ax, _pts(REF / f"{name}.ply"), _pts(REF / f"{name}_NA.ply"), f"{name}  reference")
    for i, sd in enumerate(SEEDS, start=2):
        ra, na = OUT / f"{name}_seed{sd}_RA.ply", OUT / f"{name}_seed{sd}_NA.ply"
        if not ra.exists():
            continue
        ax = fig.add_subplot(1, cols, i, projection="3d")
        _cloud_panel(ax, _pts(ra), _pts(na) if na.exists() else np.empty((0, 3)), f"seed {sd}")
    fig.suptitle(f"{name}: RA shell (grey) + NA cores (red)", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.04)
    fig.savefig(OUT / f"montage_{name}.png", dpi=140); plt.close(fig)
    print(f"wrote {OUT / f'montage_{name}.png'}")


if __name__ == "__main__":
    montage_slices()
    for nm in NAMES:
        montage_cloud(nm)
