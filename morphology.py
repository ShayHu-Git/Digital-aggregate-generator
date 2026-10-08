"""
morphology.py
=============
Particle shape descriptors used by the GUI's "reference morphology" panel
and by the target-vs-achieved comparison after the EI / FI / size rescale.

Bounding box: the oriented box aligned with the point cloud's principal
axes (PCA-OBB), the standard proxy for the minimum bounding box in
aggregate-morphology work.  Dimensions are returned sorted a >= b >= c.

    Elongation ratio  EI = b / a     (intermediate / longest)   <= 1
    Flatness   ratio  FI = c / b     (shortest / intermediate)  <= 1

(the b/a, c/b convention -- both ratios in (0, 1]).
"""
from __future__ import annotations

import numpy as np


def obb(points: np.ndarray):
    """PCA oriented bounding box.

    Returns (a, b, c, center, R) with a >= b >= c, `center` the box centre in
    world coordinates, and `R` a 3x3 matrix whose COLUMNS are the unit
    principal axes ordered (a-axis, b-axis, c-axis).  `p_local = (p-center)@R`
    puts a world point in the box frame; `p_world = p_local@R.T + center`
    brings it back.
    """
    p = np.asarray(points, dtype=np.float64)
    mean = p.mean(axis=0)
    q = p - mean
    cov = np.cov(q.T)
    evals, evecs = np.linalg.eigh(cov)          # ascending
    R = evecs[:, ::-1]                           # largest-variance axis first
    proj = q @ R
    ext = proj.max(axis=0) - proj.min(axis=0)
    order = np.argsort(ext)[::-1]                # enforce a >= b >= c
    R = np.ascontiguousarray(R[:, order])
    if np.linalg.det(R) < 0:                     # keep a right-handed frame
        R[:, 2] *= -1.0
    proj = q @ R
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    ext = hi - lo
    center = mean + R @ ((lo + hi) / 2.0)
    return float(ext[0]), float(ext[1]), float(ext[2]), center, R


def describe(mesh) -> dict:
    """Morphological parameters of one closed mesh (pv.PolyData)."""
    a, b, c, center, R = obb(mesh.points)
    v = abs(float(mesh.triangulate().volume))
    area = float(mesh.extract_surface().area)
    sph = (np.pi ** (1.0 / 3.0)) * ((6.0 * v) ** (2.0 / 3.0)) / area if area > 0 else float("nan")
    return dict(
        a=a, b=b, c=c,
        EI=b / a if a > 0 else float("nan"),      # elongation, <= 1
        FI=c / b if b > 0 else float("nan"),      # flatness,   <= 1
        volume=v, area=area, sphericity=sph,
        eq_diameter=(6.0 * v / np.pi) ** (1.0 / 3.0) if v > 0 else float("nan"),
        center=center, R=R,
    )


def describe_pair(mesh_ra, mesh_na) -> dict:
    """RA shell + (merged) NA cores together."""
    ra = describe(mesh_ra)
    out = {f"ra_{k}": v for k, v in ra.items() if k not in ("center", "R")}
    if mesh_na is not None and getattr(mesh_na, "n_points", 0) > 0:
        na = describe(mesh_na)
        out.update({f"na_{k}": v for k, v in na.items() if k not in ("center", "R")})
        # phi_NA = V_NA / V_RA  (natural-aggregate volume fraction)
        # AMC    = adhered-mortar volume / V_RA = 1 - phi_NA
        out["phi_na"] = na["volume"] / ra["volume"] if ra["volume"] > 0 else float("nan")
        out["amc"] = 1.0 - out["phi_na"] if ra["volume"] > 0 else float("nan")
    else:
        out["phi_na"] = 0.0
        out["amc"] = 1.0                       # no NA core -> the particle is all adhered mortar
    return out
