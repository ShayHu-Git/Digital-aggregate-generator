"""
synth.py
========
The lightweight generative core.

It keeps the *principle* SingleShapeGen and Sin3DM are built on -- resynthesise
a new volume from the internal statistics of a single example so the result
shares local appearance with the exemplar but differs globally -- and drops
the neural network.  No training, no GPU, deterministic given a seed,
<1 s / sample.  A diffusion / GAN backbone can replace the body of
`generate()` later without touching anything else in the pipeline.

Mechanism (single-field exemplar warp)
-------------------------------------
Working on the 2-field grid from represent.py
(channel 0 = d_RA signed distance to the shell, channel 1 = T mortar
thickness >= 0), the generator applies ONE smooth random displacement field
`W` to the whole grid:

    out = warp( [d_RA, T] , W )         W = W_low (+ W_detail)  , scaled by alpha

`W_low` is a large low-frequency field (silhouette / core-layout variation);
an optional small higher-frequency component `W_detail` adds local surface
wobble.  Because **every channel is warped by the same field**, the thin
high-value ridges in T that separate distinct NA cores stay exactly
registered with the d_RA valleys they sit in -- so `d_RA + T` cannot
develop leaks between cores.  (An earlier version warped T with its own
independent field `Wt`; that shear was the dominant cause of generated NA
cores merging -- see README.)  At `alpha = 0`, `W = 0` and `out == ref`
exactly, so the construction has no built-in volume drift.

`wrap_var` optionally varies the adhered-mortar thickness pattern by adding
a low-frequency term to T that is itself transported by `W` (rides the same
field, so it displaces nothing relative to d_RA), then T is re-clipped >= 0.
Containment (`d_NA := d_RA + clip(T,0,inf) >= d_RA`) is guaranteed
downstream regardless of anything here.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# ---- defaults (all overridable via generate(**kwargs)) ---------------------
ALPHA = 0.5
COARSE_WARP_VOX = 3.4        # peak of the low-frequency field W_low (voxels) * alpha
DETAIL_WARP_VOX = 0.6        # peak of the higher-frequency component W_detail (voxels) * alpha
WRAP_VAR = 0.0               # low-frequency mortar-thickness modulation (norm. units) * alpha
SCALE_JITTER = 0.05          # per-axis random affine scale jitter * alpha
COARSE_SMOOTH_FRAC = 0.38    # spatial smoothness of W_low  (fraction of grid size)
DETAIL_SMOOTH_FRAC = 0.12    # spatial smoothness of W_detail
WRAP_SMOOTH_FRAC = 0.32      # spatial smoothness of the wrap_var modulation


def _norm_stats(grid):
    flat = grid.reshape(-1, grid.shape[-1])
    return flat.mean(0).astype(np.float32), (flat.std(0) + 1e-6).astype(np.float32)


def _smooth_field(shape, rng, sigma_frac):
    f = rng.standard_normal(shape).astype(np.float32)
    f = gaussian_filter(f, sigma=[max(1.0, s * sigma_frac) for s in shape])
    f /= f.std() + 1e-6
    return f


def _disp(shape, rng, amp_vox, sigma_frac):
    return np.stack([_smooth_field(shape, rng, sigma_frac) for _ in range(3)],
                    axis=0) * np.float32(amp_vox)


def _scale_disp(shape, rng, amount):
    d, h, w = shape
    sc = rng.uniform(-amount, amount, size=3).astype(np.float32)
    gi, gj, gk = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cen = np.array([(d - 1) / 2, (h - 1) / 2, (w - 1) / 2], np.float32)
    return np.stack([(gi - cen[0]) * sc[0], (gj - cen[1]) * sc[1], (gk - cen[2]) * sc[2]], axis=0)


def _warp(vol, disp):
    """Displace every channel of an (D,H,W,C) volume by a (3,D,H,W) field."""
    d, h, w, c = vol.shape
    gi, gj, gk = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([gi + disp[0], gj + disp[1], gk + disp[2]], axis=0)
    out = np.empty_like(vol)
    for ch in range(c):
        out[..., ch] = map_coordinates(vol[..., ch], coords, order=1, mode="nearest")
    return out


def generate(ref_grid: np.ndarray, seed: int = 0, alpha: float = ALPHA,
             coarse_warp_vox: float = COARSE_WARP_VOX, detail_warp_vox: float = DETAIL_WARP_VOX,
             wrap_var: float = WRAP_VAR, scale_jitter: float = SCALE_JITTER,
             verbose: bool = True) -> np.ndarray:
    """ref_grid: (n,n,n,2) = [d_RA, T].  Returns a new (n,n,n,2) grid, same
    convention, with T clipped to >= 0."""
    rng = np.random.default_rng(seed)
    mean, std = _norm_stats(ref_grid)
    ref_n = ((ref_grid - mean) / std).astype(np.float32)
    shape = ref_n.shape[:3]

    # ONE displacement field for the whole grid (all channels move together)
    w = _disp(shape, rng, coarse_warp_vox * alpha, COARSE_SMOOTH_FRAC)
    w = w + _scale_disp(shape, rng, scale_jitter * alpha)
    if detail_warp_vox > 0:
        w = w + _disp(shape, rng, detail_warp_vox * alpha, DETAIL_SMOOTH_FRAC)

    out = _warp(ref_n, w)

    if wrap_var > 0:
        mod = np.stack([np.zeros(shape, np.float32),
                        _smooth_field(shape, rng, WRAP_SMOOTH_FRAC)], axis=-1)
        out[..., 1] = out[..., 1] + (wrap_var * alpha) * _warp(mod, w)[..., 1]

    out = (out * std + mean).astype(np.float32)
    out[..., 1] = np.clip(out[..., 1], 0.0, None)
    if verbose:
        occ = float((out[..., 0] < 0).mean())
        print(f"  [synth seed={seed} alpha={alpha}] shell occ={occ:.3f} "
              f"T=[{out[...,1].min():.2f},{out[...,1].max():.2f}]")
    return out
