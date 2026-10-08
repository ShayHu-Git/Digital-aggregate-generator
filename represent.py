"""
represent.py
============
Turn ONE reference RA particle (outer shell mesh + internal NA-core mesh)
into the 2-field volumetric representation the lightweight generator works
on.

Why this representation (the key change vs. an earlier two-SDF prototype)
---------------------------------------------------------
The earlier prototype stored two INDEPENDENT signed-distance channels
(d_RA, d_NA) and let the generator produce both freely.  Its honest
headline failure was that 31-37 % of the generated NA surface ended up
*outside* the RA shell -- containment was only "statistically likely",
never guaranteed, and had to be repaired by post-hoc clipping.

Here we store instead:

    channel 0 :  d_RA(x)              signed distance to the RA outer shell
    channel 1 :  T(x) = d_NA - d_RA   the "adhered-mortar thickness" field,
                                      clipped to be >= 0 everywhere

and RECONSTRUCT the NA surface as the zero level set of

    d_NA_recon(x) = d_RA(x) + T(x)          with T(x) >= 0

Because T >= 0, we have d_NA_recon(x) >= d_RA(x) for every x, hence

    { d_NA_recon < 0 }  is a subset of  { d_RA < 0 }

i.e. the NA core is inside the RA shell *by construction* -- no loss term,
no post-processing.  Where T(x) = 0 the two surfaces coincide, which is
exactly the "old mortar does not fully wrap the natural aggregate, part of
the NA is exposed at the particle surface" case the project needs.

Everything else (anisotropic per-axis grid spacing, select_enclosed_points
for the inside test, the two-EDT signed distance) is kept identical to the
spike, since those were already validated there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt

REF_DIR = Path(__file__).resolve().parent / "examples" / "ref"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# An NA "core" below this physical volume is a mortar speck / marching-cubes
# sliver, not a natural-aggregate fragment.  Core counting everywhere in the
# pipeline uses this ABSOLUTE floor, not a fraction of the total NA volume --
# a fraction-of-total threshold silently discards real small cores whenever a
# particle happens to contain many of them (this was the dominant reason the
# reported generated-core count collapsed; the N=64 grid itself keeps them).
MIN_CORE_MM3 = 0.3


def count_cores(inside: np.ndarray, spacing: np.ndarray, min_mm3: float = MIN_CORE_MM3):
    """Number of connected components of a binary `inside` mask with volume
    >= `min_mm3`.  Returns (n_significant, n_raw)."""
    from scipy.ndimage import label

    lab, n_raw = label(inside)
    if n_raw == 0:
        return 0, 0
    vox = float(np.prod(spacing))
    vols = np.bincount(lab.ravel())[1:] * vox
    return int((vols >= min_mm3).sum()), int(n_raw)


def signed_distance_field(inside_mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Approximate signed distance (world units) from a binary inside/outside
    mask via two Euclidean distance transforms.

    `spacing` is a PER-AXIS (dx, dy, dz) vector.  A real aggregate's bounding
    box is not cubic, so the physical size of a voxel genuinely differs per
    axis; scipy's `sampling=` argument is the correct way to account for that.
    (The spike found that collapsing this to one scalar inflated reconstructed
    volume by ~78 % with no model involved at all.)
    """
    dist_out = distance_transform_edt(~inside_mask, sampling=spacing)
    dist_in = distance_transform_edt(inside_mask, sampling=spacing)
    return np.where(inside_mask, -dist_in, dist_out).astype(np.float32)


def _inside_mask(query: pv.PolyData, surface: pv.PolyData, n: int) -> np.ndarray:
    enclosed = query.select_enclosed_points(surface, tolerance=1e-6, check_surface=False)
    return np.asarray(enclosed["SelectedPoints"], dtype=bool).reshape(n, n, n)




def voxelize_reference(ra_path: Path, na_path: Path, n: int = 64,
                       pad_frac: float = 0.20, k: int | None = None) -> dict:
    """Voxelize one (RA shell, NA core) mesh pair into an (n, n, n, 2) grid
    holding [d_RA, T] as described in the module docstring.

    `k` -- how many NA cores to keep, selected as the k LARGEST by volume
    from the reference.  None (default) or k >= m keeps all m reference
    cores; k = 0 keeps none (a plain mortar lump, no natural aggregate
    inside); 0 < k < m keeps the top-k.  The returned `k` / `m` /
    `kept_core_mm3` report what was actually used.

    Returns a dict with the grid plus the metadata needed to (a) map any
    generated grid back to world coordinates and (b) score generated
    samples against this reference.
    """
    from scipy.ndimage import label, map_coordinates

    ra = pv.read(str(ra_path)).clean().triangulate()
    na = pv.read(str(na_path)).clean().triangulate()

    bounds = np.array(ra.bounds, dtype=np.float64).reshape(3, 2)  # RA bounds superset NA
    center = bounds.mean(axis=1)
    half = (bounds[:, 1] - bounds[:, 0]) / 2.0 * (1.0 + pad_frac)
    lo, hi = center - half, center + half
    spacing = (hi - lo) / (n - 1)
    vox = float(np.prod(spacing))

    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    zs = np.linspace(lo[2], hi[2], n)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    query = pv.PolyData(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1))

    inside_ra = _inside_mask(query, ra, n)
    inside_na_full = _inside_mask(query, na, n) & inside_ra  # guard stray geometry

    lab, m = label(inside_na_full)
    core_vox = np.bincount(lab.ravel())[1:] if m else np.array([], dtype=int)
    order = (np.argsort(core_vox)[::-1] + 1) if m else np.array([], dtype=int)  # label ids, big first
    k_used = m if k is None else int(max(0, min(k, m)))
    keep_ids = order[:k_used].tolist()
    if k_used > 0:
        inside_na = np.isin(lab, keep_ids)
    else:
        inside_na = np.zeros_like(inside_na_full)
    kept_core_mm3 = sorted((core_vox[i - 1] * vox for i in keep_ids), reverse=True)

    d_ra = signed_distance_field(inside_ra, spacing)
    if k_used > 0:
        d_na = signed_distance_field(inside_na, spacing)
        thickness = np.clip(d_na - d_ra, 0.0, None).astype(np.float32)  # T >= 0 by clip
    else:  # no NA core -> push d_NA fully positive so no zero crossing exists
        thickness = np.full_like(d_ra, float(np.abs(d_ra).max() * 2.0 + 10.0))

    n_cores_signif, n_cores_raw = count_cores(inside_na, spacing)
    grid = np.stack([d_ra, thickness], axis=-1).astype(np.float32)

    # reference's own "exposed NA" fraction (natural aggregate reaching the
    # particle surface); computed on the FULL reference NA regardless of k.
    tol = 0.5 * float(np.linalg.norm(spacing))
    if m:
        t_full = np.clip(signed_distance_field(inside_na_full, spacing) - d_ra, 0.0, None)
        gc = (na.points - lo[None, :]) / spacing[None, :]
        exposed_frac_ref = float(np.mean(map_coordinates(t_full, gc.T, order=1, mode="nearest") <= tol))
    else:
        exposed_frac_ref = 0.0

    return dict(
        grid=grid,
        origin=lo.astype(np.float32),
        spacing=spacing.astype(np.float32),
        n=n,
        ra_volume=float(ra.volume),
        na_volume=float(na.volume),
        # phi_NA = V_NA / V_RA ; adhered-mortar content AMC = 1 - phi_NA
        na_frac=float(na.volume / ra.volume),
        amc_proxy=float(1.0 - na.volume / ra.volume),
        n_cores_ref=n_cores_signif,
        n_cores_ref_raw=n_cores_raw,
        m=int(m), k=k_used, kept_core_mm3=[float(x) for x in kept_core_mm3],
        frac_inside_ra=float(inside_ra.mean()),
        frac_inside_na=float(inside_na.mean()),
        exposed_frac_ref=exposed_frac_ref,
        exposed_tol=tol,
    )


# backwards-compatible alias for the GUI / pipeline
voxelize_pair = voxelize_reference


def build_reference(name: str, n: int = 64, force: bool = False) -> Path:
    """Voxelize ref/<name>.ply + ref/<name>_NA.ply, cache to data/<name>_N<n>.npz,
    return the cache path."""
    out = DATA_DIR / f"{name}_N{n}.npz"
    if out.exists() and not force:
        return out
    res = voxelize_reference(REF_DIR / f"{name}.ply", REF_DIR / f"{name}_NA.ply", n=n)
    meta = {k: v for k, v in res.items() if k not in ("grid", "origin", "spacing")}
    np.savez(out, grid=res["grid"], origin=res["origin"], spacing=res["spacing"],
             meta=json.dumps(meta))
    print(f"[represent] {name}: N={n} RA_vol={res['ra_volume']:.1f} "
          f"NA_vol={res['na_volume']:.1f} AMC~={res['amc_proxy']:.3f} "
          f"ref_cores={res['n_cores_ref']} (raw {res['n_cores_ref_raw']}) "
          f"exposed_frac={res['exposed_frac_ref']:.3f} spacing={np.round(res['spacing'], 3)}")
    return out


def load_reference(path: Path):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return d["grid"], d["origin"], d["spacing"], meta


if __name__ == "__main__":
    for nm in ("A2", "A7", "A8"):
        build_reference(nm, n=64, force=True)
