"""
paper_figures.py
================
Publication figures for the RA-particle generator: the encoding of the
reference particle A2 on one whole cross-section (Figure 2, panels e-g), and
the SAME method applied to one particle A2' generated from A2 (Figure 3,
panels e-g), so the two can be compared directly.

Every panel is saved as its OWN single-plot PNG (no composite / subplot
grids) so it can be dropped straight into a manuscript.

Style: scienceplots  ["science"]  -- that style sheet is
the single source of truth for figure size, fonts, grid, colours and the
LaTeX text backend.  `--no-latex` swaps in scienceplots' "no-latex" (mathtext).

    python paper_figures.py                # every panel -> figs/
    python paper_figures.py --only 1       # Figure 2 (e-g): reference A2
    python paper_figures.py --only 2       # Figure 3 (e-g): A2' generated from A2
    python paper_figures.py --no-latex     # skip the LaTeX text backend (fast)

Output (dpi from science.mplstyle, in figs/):
    fig2a_voxels  fig2b_dRA  fig2c_tau     (reference A2, whole slice)
    fig3a_voxels  fig3b_dRA  fig3c_tau     (generated A2', SAME section/method)
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import scienceplots  # noqa: F401  (registers the "science" family)

import represent
import synth

HERE = Path(__file__).resolve().parent
REF = represent.REF_DIR
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
RA_PLY, NA_PLY = str(REF / "A2.ply"), str(REF / "A2_NA.ply")

# phase colours (consistent across every panel):
#   mortar shell = tan, natural-aggregate core = teal
C_MORTAR = "#d9c8a9"
C_NA = "#8db0b6"
C_OUTLINE = "#2b2b2b"
PHASE_CMAP = ListedColormap(["white", C_MORTAR, C_NA])

# one particle generated from A2, used throughout Figure 3.  alpha=0.7 and
# this seed were chosen (from a seed/alpha sweep at the SAME n=34 resolution
# used below) so the generated section is clearly distinguishable from the
# reference at a glance, while alpha stays inside the range validated in the
# report (IoU ~= 0.85/0.76/0.68 at alpha = 0.3/0.5/0.7).
GEN_SEED = 1
GEN_ALPHA = 0.3

# resolution shared by every panel below: both the reference and the
# generated particle are voxelised at the SAME n, so their "blockiness" and
# their section are directly comparable, not an artefact of different grids.
N_SLICE = 34


# --------------------------------------------------------------------------- #
#  style  --  every panel uses scienceplots "science" (that file is the style)
# --------------------------------------------------------------------------- #
# `import scienceplots` (above) registers "science" / "no-latex" in memory.
# The chain is resolved + applied once here (import time) and re-applied around
# every builder by `styled`, so the science style is in force no matter how a
# builder is called (script entry point, notebook, REPL, ...).
_LATEX_OK: bool | None = None


def _latex_ok() -> bool:
    """True once the "science" LaTeX text backend is confirmed to render."""
    global _LATEX_OK
    if _LATEX_OK is None:
        try:
            with plt.style.context(["science"]):
                fig = plt.figure()
                fig.text(0.5, 0.5, r"$d_{\mathrm{RA}}$")
                fig.canvas.draw()
                plt.close(fig)
            _LATEX_OK = True
        except Exception as exc:  # noqa: BLE001
            print(f"[style] LaTeX backend unavailable ({exc}); using science + no-latex")
            _LATEX_OK = False
    return _LATEX_OK


def _science_chain(no_latex: bool) -> list[str]:
    """["science"] (+ "no-latex" when the LaTeX backend is unwanted or unusable)."""
    no_latex = no_latex or not _latex_ok()
    return ["science", "no-latex"] if no_latex else ["science"]


STYLE = _science_chain(no_latex=False)
plt.style.use(STYLE)


def apply_style(no_latex: bool = False) -> None:
    """Re-select the figure style; pass no_latex=True for the fast (no-LaTeX) path."""
    global STYLE
    STYLE = _science_chain(no_latex)
    plt.style.use(STYLE)


def styled(fn):
    """Draw a figure builder inside the resolved scienceplots ["science", ...] style."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.style.context(STYLE):
            return fn(*args, **kwargs)
    return wrapper


def _save(fig, name):
    plt.savefig(FIGS / f"{name}.png")        # dpi comes from savefig.dpi in science.mplstyle
    plt.close(fig)
    print(f"{name}.png")


def _slice_index(d_na):
    ins = d_na < 0
    return int(round(np.argwhere(ins)[:, 0].mean())) if ins.any() else d_na.shape[0] // 2


# --------------------------------------------------------------------------- #
#  the shared method: voxelisation / d_RA field / tau field on ONE whole
#  cross-section.  Called once for the reference (Figure 2 e-g) and once for
#  a generated particle A2' (Figure 3 e-g) so the two use an identical
#  visual grammar and are directly comparable.
# --------------------------------------------------------------------------- #
def _slice_panels(d_ra, tau, d_na, ue, ve, outline, file_prefix, prime=False):
    """d_ra, tau, d_na: the three fields on one section (2-D arrays).
    ue, ve: cell-edge (u, v) coordinates [mm] for the pcolormesh grid.
    outline = (o_ra, o_na, ou, ov): the field(s) + cell-CENTRE coordinates
    used to draw the RA/NA surface contour lines over the blocky cells.
    file_prefix: "fig2" (reference) or "fig3" (generated).
    prime: label the reference's own surfaces with a trailing ' (A2')."""
    o_ra, o_na, ou, ov = outline
    tag = "'" if prime else ""

    phase = np.zeros_like(d_ra, dtype=int)
    phase[d_ra < 0] = 1                            # adhered mortar
    phase[d_na < 0] = 2                            # natural aggregate

    def _finish(ax):
        """Surface outlines + axes clamped exactly to the voxel-grid edges."""
        ax.grid(False)
        ax.contour(ov, ou, o_ra, [0.0], colors=C_OUTLINE, linewidths=1.3)
        ax.contour(ov, ou, o_na, [0.0], colors="#136b73", linewidths=1.3)
        ax.set_xlim(ve[0], ve[-1])
        ax.set_ylim(ue[0], ue[-1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")

    # ---------- (a) voxelisation --------------------------------------- #
    fig, ax = plt.subplots()
    ax.pcolormesh(ve, ue, phase, cmap=PHASE_CMAP, vmin=0, vmax=2,
                  edgecolors="#c2c2c2", linewidth=0.3)
    _finish(ax)
    ax.legend(handles=[Patch(fc=C_MORTAR, ec="#b8a888", label="AM"),
                       Patch(fc=C_NA, ec="#6f9199", label="NA"),
                       Line2D([0], [0], color=C_OUTLINE, lw=1.3, label=f"RA{tag} surface"),
                       Line2D([0], [0], color="#136b73", lw=1.3, label=f"NA{tag} surface")],
              loc="center", ncol=2, frameon=False, fancybox=False, bbox_to_anchor=(0.5, -0.15),
              handlelength=1.3, columnspacing=1.1)
    _save(fig, f"{file_prefix}a_voxels")

    # ---------- (b) d_RA field --------------------------------------- #
    fig, ax = plt.subplots()
    lim = 8.0
    pcm = ax.pcolormesh(ve, ue, np.clip(d_ra, -lim, lim), cmap="coolwarm",
                        vmin=-lim, vmax=lim, edgecolors="#cfcfcf", linewidth=0.22)
    _finish(ax)
    cb = fig.colorbar(pcm, ax=ax, orientation="horizontal", location="bottom",
                      fraction=0.046, shrink=1.0, pad=0.03, ticks=np.linspace(-lim, lim, 5))
    cb.set_label(rf"$d_{{\mathrm{{RA}}}}{tag}$  [mm]")
    _save(fig, f"{file_prefix}b_dRA")

    # ---------- (c) tau field -------------------------------------- #
    fig, ax = plt.subplots()
    tmax = float(np.ceil(np.nanmax(np.where(d_ra < 0, tau, np.nan))))
    pcm = ax.pcolormesh(ve, ue, np.where(d_ra < 0, tau, np.nan), cmap="coolwarm",
                        vmin=0.0, vmax=tmax, edgecolors="#cfcfcf", linewidth=0.22)
    _finish(ax)
    cb = fig.colorbar(pcm, ax=ax, orientation="horizontal", location="bottom",
                      fraction=0.046, shrink=1.0, pad=0.03, ticks=np.arange(0, tmax + 0.1, 2.0))
    cb.set_label(rf"$\tau{tag}$  [mm]")
    _save(fig, f"{file_prefix}c_tau")


def _a2_slice(n):
    """One axis-0 cross-section of the A2 encoding, taken through the NA cores.
    Returns d_RA, tau, d_NA on the slice and the world (mm) cell-centre (u, v)
    and cell-edge (ue, ve) coordinates of the two in-plane axes
    (v = grid axis 2, u = grid axis 1)."""
    res = represent.voxelize_pair(RA_PLY, NA_PLY, n=n, k=None)
    g, sp, org = res["grid"], res["spacing"], res["origin"]
    d_ra = g[..., 0]
    tau = np.clip(g[..., 1], 0.0, None)
    d_na = d_ra + tau
    z = _slice_index(d_na)
    u = org[1] + np.arange(d_ra.shape[1]) * sp[1]
    v = org[2] + np.arange(d_ra.shape[2]) * sp[2]
    ue = np.r_[u - sp[1] / 2, u[-1] + sp[1] / 2]
    ve = np.r_[v - sp[2] / 2, v[-1] + sp[2] / 2]
    return dict(d_ra=d_ra[z], tau=tau[z], d_na=d_na[z], u=u, v=v, ue=ue, ve=ve,
                du=float(sp[1]), dv=float(sp[2]), x_phys=float(org[0] + z * sp[0]))


def _fine_slice(x_phys, n=150):
    """High-resolution d_RA / d_NA on the same plane x = x_phys, for drawing
    the smooth reference phase outlines the voxel cells approximate.  Only
    meaningful for the reference: it re-voxelises the ORIGINAL X-CT mesh at
    high resolution, which is the true, independent ground truth for the
    surfaces the coarse grid samples."""
    res = represent.voxelize_pair(RA_PLY, NA_PLY, n=n, k=None)
    g, sp, org = res["grid"], res["spacing"], res["origin"]
    zf = int(np.clip(round((x_phys - org[0]) / sp[0]), 0, g.shape[0] - 1))
    d_ra = g[zf, :, :, 0]
    d_na = d_ra + np.clip(g[zf, :, :, 1], 0.0, None)
    u = org[1] + np.arange(d_ra.shape[0]) * sp[1]
    v = org[2] + np.arange(d_ra.shape[1]) * sp[2]
    return d_ra, d_na, u, v


@styled
def fig_encoding_slice():
    """Figure 2 (e-g): the encoding on ONE full cross-section of the A2
    reference -- (e) voxelisation, (f) the d_RA field, (g) the tau field.
    The surface outlines come from a separate, high-resolution voxelisation
    of the ORIGINAL X-CT mesh (the true smooth ground truth the coarse grid
    approximates)."""
    S = _a2_slice(n=N_SLICE)
    fr_ra, fr_na, fu, fv = _fine_slice(S["x_phys"])
    _slice_panels(S["d_ra"], S["tau"], S["d_na"], S["ue"], S["ve"],
                 outline=(fr_ra, fr_na, fu, fv), file_prefix="fig2", prime=False)


@styled
def fig_encoding_slice_generated():
    """Figure 3 (e-g): the SAME method as Figure 2 (e-g), applied to ONE
    particle A2' generated from the A2 reference (Section "Making new
    particles"), on the SAME section and at the SAME voxel resolution, so the
    two are directly comparable. Because A2' only exists as this warped
    field (there is no independent fine mesh for it, unlike the reference),
    its surface outlines are drawn from its OWN coarse field rather than from
    a separate high-resolution source."""
    res = represent.voxelize_pair(RA_PLY, NA_PLY, n=N_SLICE, k=None)
    grid, sp, org = res["grid"], res["spacing"], res["origin"]
    d_ra_ref = grid[..., 0]
    d_na_ref = d_ra_ref + np.clip(grid[..., 1], 0.0, None)
    z = _slice_index(d_na_ref)                      # the same section as Figure 2 (e-g)

    g = synth.generate(grid, seed=GEN_SEED, alpha=GEN_ALPHA, verbose=True)
    d_ra = g[z, ..., 0]
    tau = np.clip(g[z, ..., 1], 0.0, None)
    d_na = d_ra + tau

    u = org[1] + np.arange(d_ra.shape[0]) * sp[1]
    v = org[2] + np.arange(d_ra.shape[1]) * sp[2]
    ue = np.r_[u - sp[1] / 2, u[-1] + sp[1] / 2]
    ve = np.r_[v - sp[2] / 2, v[-1] + sp[2] / 2]

    _slice_panels(d_ra, tau, d_na, ue, ve, outline=(d_ra, d_na, u, v),
                 file_prefix="fig3", prime=True)


_BUILDERS = ["fig_encoding_slice", "fig_encoding_slice_generated"]
ALL = {i: globals()[name] for i, name in enumerate(_BUILDERS, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, choices=list(ALL), help="build one group")
    ap.add_argument("--no-latex", action="store_true", help="skip the LaTeX text backend")
    args = ap.parse_args()
    apply_style(args.no_latex)          # STYLE is already resolved at import; this
                                        # only re-applies it / honours --no-latex
    for i, fn in ALL.items():
        if args.only and i != args.only:
            continue
        fn()
    print("panels ->", FIGS)


if __name__ == "__main__":
    main()
