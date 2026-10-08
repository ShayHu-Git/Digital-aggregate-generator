"""
run.py
======
End-to-end feasibility / stability run.

For each reference particle (A2, A7, A8):
  1. voxelise -> [d_RA, T] grid                                (represent.py)
  2. encode->decode roundtrip check (no generator)             (reconstruct.py)
  3. for K seeds: synthesise a new grid                        (synth.py)
       -> reconstruct RA shell + merged NA cores (field-space  (reconstruct.py)
          Gaussian smoothing, containment-preserving)
       -> drop marching-cubes speckle on the NA side           (postprocess.py)
       -> write  out/<name>_seed<k>_RA.ply   (outer RA shell)
                 out/<name>_seed<k>_NA.ply   (all NA cores merged, one file)
       -> score it                                             (metrics.py)
  4. write out/report.json and out/report.md

Run:  python run.py            # full: A2 A7 A8, 4 seeds, N=64
      python run.py --quick    # A8 only, 2 seeds, N=48   (smoke test)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import represent
import reconstruct
import postprocess
import metrics
import synth

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def _vol(m):
    return metrics.mesh_volume(m)


def roundtrip(grid, origin, spacing, meta, smooth_sigma):
    m_ra, m_na, _ = reconstruct.grid_to_meshes(grid, origin, spacing, margin=0.0,
                                               smooth_sigma=smooth_sigma)
    return dict(ra_dpct=metrics.pct_delta(_vol(m_ra), meta["ra_volume"]),
                na_dpct=metrics.pct_delta(_vol(m_na), meta["na_volume"]))


def process_seed(name, seed, grid_ref, origin, spacing, meta, alpha, smooth_sigma,
                 volume_match=True, margin=0.0, min_core_mm3=represent.MIN_CORE_MM3):
    t0 = time.time()
    g = synth.generate(grid_ref, seed=seed, alpha=alpha, verbose=True)
    m_ra, m_na, d_ra_s = reconstruct.grid_to_meshes(g, origin, spacing, margin=margin,
                                                    smooth_sigma=smooth_sigma)
    if m_ra is None:
        print(f"  [{name} seed{seed}] RA channel had no zero crossing -- skipped")
        return None, None
    m_ra = m_ra.triangulate().clean()
    n_cores_kept = 0
    if m_na is not None and m_na.n_points:
        m_na, n_cores_kept = postprocess.filter_small_components(m_na, min_volume=min_core_mm3)

    # containment + wrapping measured BEFORE the volume match (that step is a
    # similarity transform about the RA centroid, which provably preserves
    # nesting and surface-touching, so measuring it here is valid)
    contain_sdf = metrics.containment_sdf(m_na, d_ra_s, origin, spacing)
    exposed = metrics.exposed_fraction(m_na, g[..., 1], origin, spacing, meta["exposed_tol"])
    ra_dpct_raw = metrics.pct_delta(_vol(m_ra), meta["ra_volume"])

    scale = 1.0
    if volume_match:
        m_ra, m_na, scale = postprocess.match_ra_volume(m_ra, m_na, meta["ra_volume"])

    stem = f"{name}_seed{seed}"
    m_ra.save(OUT / f"{stem}_RA.ply")
    if m_na is not None and m_na.n_points:
        m_na.save(OUT / f"{stem}_NA.ply")

    _, d_na_s = reconstruct.smoothed_fields(g, margin=margin, smooth_sigma=smooth_sigma)
    n_grid = postprocess.count_cores(d_na_s, spacing, min_mm3=min_core_mm3)

    ra_v, na_v = _vol(m_ra), _vol(m_na)
    rec = dict(
        name=name, seed=seed, alpha=alpha, scale=round(scale, 4), ra_dpct_raw=ra_dpct_raw,
        ra_dpct=metrics.pct_delta(ra_v, meta["ra_volume"]),
        na_dpct=metrics.pct_delta(na_v, meta["na_volume"]),
        amc=(1.0 - na_v / ra_v) if ra_v else float("nan"),  # adhered-mortar content = 1 - V_NA/V_RA
        amc_ref=meta["amc_proxy"],
        n_cores=n_grid, n_cores_mesh=n_cores_kept, n_cores_ref=meta["n_cores_ref"],
        n_cores_ref_raw=meta.get("n_cores_ref_raw"),
        exposed_frac=exposed, exposed_frac_ref=meta["exposed_frac_ref"],
        contain_sdf_outside=contain_sdf,
        contain_encl_outside=metrics.containment_enclosed(m_na, m_ra),
        seconds=round(time.time() - t0, 1),
    )
    print(f"  [{name} seed{seed}] RA {rec['ra_dpct']:+.1f}% (raw {ra_dpct_raw:+.1f}%)  "
          f"NA {rec['na_dpct']:+.1f}%  AMC {rec['amc']:.3f}/{rec['amc_ref']:.3f}  "
          f"cores {n_grid} (ref {meta['n_cores_ref']})  "
          f"exposed {exposed:.2f}/{meta['exposed_frac_ref']:.2f}  "
          f"NA-out sdf={rec['contain_sdf_outside']*100:.2f}% "
          f"encl={rec['contain_encl_outside']*100:.2f}%  ({rec['seconds']}s)")
    return rec, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--smooth-sigma", type=float, default=0.6)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="world-unit gap forced between the NA cores and the RA shell "
                         "(0 = allow the surfaces to touch, i.e. exposed natural aggregate)")
    ap.add_argument("--no-volume-match", action="store_true",
                    help="skip the analytic similarity-scale to the reference RA volume")
    ap.add_argument("--min-core-mm3", type=float, default=represent.MIN_CORE_MM3,
                    help="absolute volume floor below which an NA component is a speck")
    args = ap.parse_args()

    names = args.names or (["A8"] if args.quick else ["A2", "A7", "A8"])
    k = args.seeds or (2 if args.quick else 4)
    n = args.n or (48 if args.quick else 64)
    seeds = list(range(1, k + 1))

    all_recs, rt = [], {}
    for name in names:
        path = represent.build_reference(name, n=n)
        grid_ref, origin, spacing, meta = represent.load_reference(path)
        rt[name] = roundtrip(grid_ref, origin, spacing, meta, args.smooth_sigma)
        print(f"[roundtrip {name}] RA {rt[name]['ra_dpct']:+.1f}%  NA {rt[name]['na_dpct']:+.1f}%  "
              f"(ref cores={meta['n_cores_ref']} raw {meta.get('n_cores_ref_raw')}, "
              f"AMC~{meta['amc_proxy']:.3f})")

        grids = []
        for s in seeds:
            rec, g = process_seed(name, s, grid_ref, origin, spacing, meta,
                                  args.alpha, args.smooth_sigma,
                                  volume_match=not args.no_volume_match, margin=args.margin,
                                  min_core_mm3=args.min_core_mm3)
            if rec:
                all_recs.append(rec)
                grids.append(g)
        div = metrics.pairwise_iou(grids)
        for r in all_recs:
            if r["name"] == name and "pairwise_iou_shell" not in r:
                r["pairwise_iou_shell"] = div
        print(f"[diversity {name}] mean pairwise shell IoU across seeds = {div:.3f}")

    report = dict(config=dict(names=names, seeds=seeds, n=n, alpha=args.alpha,
                              smooth_sigma=args.smooth_sigma, margin=args.margin,
                              min_core_mm3=args.min_core_mm3,
                              volume_match=not args.no_volume_match),
                  roundtrip=rt, samples=all_recs)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_md(report)
    print(f"\nwrote {OUT/'report.json'} and {OUT/'report.md'}")


def _write_md(report):
    L = ["# ra_diffusion_lite -- feasibility / stability report", "",
         f"`config = {report['config']}`", "",
         "## 1. Encode->decode roundtrip (no generator; grid-resolution ceiling)", "",
         "| ref | RA vol dpct | NA vol dpct |", "|---|---|---|"]
    for nm, r in report["roundtrip"].items():
        L.append(f"| {nm} | {r['ra_dpct']:+.1f}% | {r['na_dpct']:+.1f}% |")
    L += ["", "## 2. Generated samples", "",
          "| ref | seed | RA dpct (raw) | NA dpct | AMC / ref | cores / ref (raw) | "
          "exposed / ref | NA outside RA  sdf / encl | shell IoU | s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in report["samples"]:
        L.append(
            f"| {s['name']} | {s['seed']} | {s['ra_dpct']:+.1f}% ({s.get('ra_dpct_raw', float('nan')):+.1f}%) "
            f"| {s['na_dpct']:+.1f}% | "
            f"{s['amc']:.3f} / {s['amc_ref']:.3f} | "
            f"{s.get('n_cores', '?')} / {s['n_cores_ref']} ({s.get('n_cores_ref_raw', '?')}) | "
            f"{s.get('exposed_frac', float('nan')):.2f} / {s.get('exposed_frac_ref', float('nan')):.2f} | "
            f"{s['contain_sdf_outside']*100:.2f}% / {s['contain_encl_outside']*100:.2f}% | "
            f"{s.get('pairwise_iou_shell', float('nan')):.3f} | {s['seconds']} |")
    L += ["", "## 3. How to read it", "",
          "- **RA dpct (raw)**: volume error after the analytic similarity-scale to "
          "the reference RA volume; the parenthesised raw value is before that step "
          "(the random warp adds a mild, consistent net volume drift).",
          "- **NA outside RA / sdf**: NA vertices sampled against the generated d_RA "
          "field. 0.00% is the structural guarantee of the thickness-field "
          "representation -- no clipping is applied anywhere.",
          "- **NA outside RA / encl**: independent ray-cast (`select_enclosed_points`). "
          "Small residuals are coincident-surface noise where a core is meant to touch "
          "the shell, not real penetration.",
          "- **shell IoU**: mean pairwise IoU of the `{d_RA<0}` masks across seeds. "
          "Below 1.0 = variants are not identical; well above 0 = they still share the "
          "reference's gestalt.",
          "- **cores / ref (raw)**: connected components of the generated `{d_NA<0}` "
          "field with volume >= min_core_mm3, vs the reference's significant / raw "
          "core count. Two fixes vs the earlier version: (1) input side -- count with "
          "an ABSOLUTE volume floor, not a fraction of total NA volume (the N=64 grid "
          "keeps the septa; a fraction-of-total threshold was discarding real small "
          "cores); (2) output side -- the thickness channel T is now warped by the "
          "SAME displacement field as d_RA (no independent `Wt`), so the mortar-vein "
          "ridges that separate cores stay registered with the d_RA valleys instead of "
          "shearing off them. The exported .ply still merges all cores into one file.",
          "- **exposed / ref**: fraction of NA surface with mortar thickness ~0 "
          "(natural aggregate exposed at the particle surface) vs the reference's own "
          "value -- the partial- vs full-wrapping control."]
    (OUT / "report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
