"""
pipeline.py
===========
One place that runs the whole thing for the GUI (and for scripts):

    analyze_reference(ra_ply, na_ply)      -> reference morphology + NA-core sizes
    generate(params, progress=...)         -> list of (mesh_RA, mesh_NA, info) + report

Parameters (see `GenParams`):
    k            NA cores to keep, in [0, m]  (default m = reference count)
    alpha        deviation-from-reference knob in [0, 1]
    n_particles  how many particles to generate (l)
    custom_shape if True, rescale each particle to target EI / FI / size band
    target_EI    elongation ratio b/a   (in (0, 1]; default = reference)
    target_FI    flatness   ratio c/b   (in (0, 1]; default = reference)
    size_min/max intermediate box dimension b is scaled into this band
    N, smooth_sigma, margin, seed_base   advanced / reproducibility
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

import represent
import synth
import reconstruct
import postprocess
import metrics
import morphology


@dataclass
class GenParams:
    ra_ply: str
    na_ply: str
    out_dir: str = "out"
    k: int | None = None
    alpha: float = 0.5
    n_particles: int = 1
    custom_shape: bool = False
    target_EI: float | None = None
    target_FI: float | None = None
    size_min: float | None = None
    size_max: float | None = None
    N: int = 64
    smooth_sigma: float = 0.6
    margin: float = 0.0
    seed_base: int = 1
    min_core_mm3: float = represent.MIN_CORE_MM3


def analyze_reference(ra_ply: str, na_ply: str) -> dict:
    """Morphology of the reference RA shell + NA cores, and each NA core's
    volume (largest first) so the GUI can show what k will select."""
    ra = pv.read(ra_ply).clean().triangulate()
    na = pv.read(na_ply).clean().triangulate()
    ra_d = morphology.describe(ra)
    na_d = morphology.describe(na)
    core_vols = []
    try:
        for b in na.connectivity("all").split_bodies():
            s = b.extract_surface().triangulate().clean()
            if s.n_points:
                core_vols.append(abs(float(s.volume)))
    except Exception:
        core_vols = [abs(float(na.volume))]
    core_vols.sort(reverse=True)
    phi_na = na_d["volume"] / ra_d["volume"] if ra_d["volume"] > 0 else float("nan")
    return dict(
        ra=ra_d, na=na_d,
        phi_na=phi_na,                          # natural-aggregate volume fraction V_NA / V_RA
        amc=1.0 - phi_na,                       # adhered-mortar content = 1 - V_NA / V_RA
        m=len(core_vols), core_volumes=core_vols,
    )


def _one_particle(grid, origin, spacing, p: GenParams, seed, ref_ei, ref_fi, rng):
    t0 = time.time()
    g = synth.generate(grid, seed=seed, alpha=p.alpha, verbose=False)
    m_ra, m_na, d_ra_s = reconstruct.grid_to_meshes(g, origin, spacing, margin=p.margin,
                                                    smooth_sigma=p.smooth_sigma)
    if m_ra is None:
        return None
    m_ra = m_ra.triangulate().clean()
    n_cores = 0
    if m_na is not None and m_na.n_points:
        m_na, n_cores = postprocess.filter_small_components(m_na, min_volume=p.min_core_mm3)

    contain = metrics.containment_sdf(m_na, d_ra_s, origin, spacing)  # before any rescale

    shape_cmp = {}
    if p.custom_shape:
        smin = p.size_min if p.size_min is not None else morphology.describe(m_ra)["b"]
        smax = p.size_max if p.size_max is not None else smin
        b_t = float(rng.uniform(min(smin, smax), max(smin, smax))) if smax != smin else float(smin)
        m_ra, m_na, shape_cmp = postprocess.rescale_to_shape(
            m_ra, m_na, p.target_EI or ref_ei, p.target_FI or ref_fi, b_t)

    d = morphology.describe(m_ra)
    na_vol = abs(float(m_na.triangulate().volume)) if (m_na is not None and m_na.n_points) else 0.0
    info = dict(
        seed=seed, a=d["a"], b=d["b"], c=d["c"], EI=d["EI"], FI=d["FI"],
        volume=d["volume"], area=d["area"], sphericity=d["sphericity"],
        n_cores=n_cores,
        amc=(1.0 - na_vol / d["volume"]) if d["volume"] > 0 else float("nan"),  # 1 - V_NA/V_RA
        contain_sdf_outside=contain, shape_cmp=shape_cmp,
        seconds=round(time.time() - t0, 2),
    )
    return m_ra, m_na, info


def generate(p: GenParams, progress=lambda s: None) -> dict:
    progress(f"voxelising reference (N={p.N}, k={'all' if p.k is None else p.k}) ...")
    res = represent.voxelize_pair(p.ra_ply, p.na_ply, n=p.N, k=p.k)
    grid, origin, spacing = res["grid"], res["origin"], res["spacing"]
    ref_ra = morphology.describe(pv.read(p.ra_ply).clean().triangulate())
    ref_ei, ref_fi = ref_ra["EI"], ref_ra["FI"]
    progress(f"reference: m={res['m']} cores, using k={res['k']}; "
             f"kept core volumes (mm^3) = {[round(v,2) for v in res['kept_core_mm3']]}")

    particles, rng = [], np.random.default_rng(p.seed_base * 104729)
    for i in range(p.n_particles):
        seed = p.seed_base + i
        out = _one_particle(grid, origin, spacing, p, seed, ref_ei, ref_fi, rng)
        if out is None:
            progress(f"  particle {i + 1}/{p.n_particles}: RA channel empty -- skipped")
            continue
        _, _, info = out
        particles.append(out)
        msg = (f"  particle {i + 1}/{p.n_particles} seed={seed}: "
               f"EI={info['EI']:.3f} FI={info['FI']:.3f} b={info['b']:.2f} "
               f"V={info['volume']:.1f} cores={info['n_cores']} "
               f"NA-outside={info['contain_sdf_outside'] * 100:.2f}% ({info['seconds']}s)")
        progress(msg)

    report = _report_md(p, res, ref_ra, [info for _, _, info in particles])
    return dict(ref=res, ref_morph=ref_ra, particles=particles, report=report)


def save_particles(result: dict, out_dir: str, stem: str) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (m_ra, m_na, info) in enumerate(result["particles"], start=1):
        ra_p = str(Path(out_dir) / f"{stem}_p{i:02d}_RA.ply")
        m_ra.save(ra_p)
        paths.append(ra_p)
        if m_na is not None and m_na.n_points:
            na_p = str(Path(out_dir) / f"{stem}_p{i:02d}_NA.ply")
            m_na.save(na_p)
            paths.append(na_p)
    (Path(out_dir) / f"{stem}_report.md").write_text(result["report"], encoding="utf-8")
    paths.append(str(Path(out_dir) / f"{stem}_report.md"))
    return paths


def _report_md(p: GenParams, res: dict, ref: dict, infos: list[dict]) -> str:
    L = ["# ra_diffusion_lite -- generation report", ""]
    L += [f"- reference RA : `{p.ra_ply}`",
          f"- reference NA : `{p.na_ply}`",
          f"- params: alpha={p.alpha}  k={res['k']}/{res['m']}  particles={p.n_particles}  "
          f"N={p.N}  smooth_sigma={p.smooth_sigma}  margin={p.margin}  "
          f"custom_shape={p.custom_shape}", ""]
    L += ["## Reference morphology (RA shell)", "",
          "| a [mm] | b [mm] | c [mm] | EI = b/a | FI = c/b | Volume [mm3] | Area [mm2] | "
          "Sphericity | NA cores m | Adhered mortar content [%] |",
          "|---|---|---|---|---|---|---|---|---|---|",
          f"| {ref['a']:.2f} | {ref['b']:.2f} | {ref['c']:.2f} | {ref['EI']:.3f} | "
          f"{ref['FI']:.3f} | {ref['volume']:.1f} | {ref['area']:.1f} | "
          f"{ref['sphericity']:.3f} | {res['m']} | {res['amc_proxy'] * 100:.1f} |", ""]
    if res["kept_core_mm3"]:
        L += [f"NA cores kept (largest {res['k']} of {res['m']}), volume mm^3: "
              f"{[round(v, 2) for v in res['kept_core_mm3']]}", ""]

    L += ["## Generated particles", "",
          "| # | seed | a [mm] | b [mm] | c [mm] | EI=b/a | FI=c/b | Volume [mm3] | "
          "Sphericity | cores | AMC [%] | NA outside RA |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(infos, start=1):
        L.append(
            f"| {i} | {s['seed']} | {s['a']:.2f} | {s['b']:.2f} | {s['c']:.2f} | "
            f"{s['EI']:.3f} | {s['FI']:.3f} | {s['volume']:.1f} | {s['sphericity']:.3f} | "
            f"{s['n_cores']} | {s['amc'] * 100:.1f} | {s['contain_sdf_outside'] * 100:.2f}% |")

    if p.custom_shape and infos and infos[0].get("shape_cmp"):
        L += ["", "## Target vs achieved morphology", "",
              "| # | EI target/achieved (err) | FI target/achieved (err) | "
              "b target/achieved (err) |", "|---|---|---|---|"]
        for i, s in enumerate(infos, start=1):
            c = s["shape_cmp"]
            if not c:
                continue
            ei_e = abs(c["achieved_EI"] - c["target_EI"]) / c["target_EI"] * 100
            fi_e = abs(c["achieved_FI"] - c["target_FI"]) / c["target_FI"] * 100
            b_e = abs(c["achieved_b"] - c["target_b"]) / c["target_b"] * 100
            L.append(f"| {i} | {c['target_EI']:.3f} / {c['achieved_EI']:.3f} ({ei_e:.1f}%) | "
                     f"{c['target_FI']:.3f} / {c['achieved_FI']:.3f} ({fi_e:.1f}%) | "
                     f"{c['target_b']:.2f} / {c['achieved_b']:.2f} ({b_e:.1f}%) |")

    L += ["", "## Notes", "",
          "- **NA outside RA**: fraction of NA vertices outside the generated d_RA "
          "field, measured before the shape rescale. 0.00% is the structural "
          "guarantee of the d_NA = d_RA + T (T>=0) representation.",
          "- The shape rescale is one affine map applied to the RA shell and every "
          "NA core together, so containment is preserved through it.",
          "- k selects the k largest reference NA cores as templates; k=0 -> a plain "
          "mortar particle with no natural-aggregate core."]
    return "\n".join(L)
