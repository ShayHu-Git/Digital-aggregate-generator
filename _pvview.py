"""Standalone interactive PyVista viewer, launched as a subprocess by gui.py
so the GUI's own event loop is never blocked.

    python _pvview.py <RA.ply> [<NA.ply>]
"""
import sys

import pyvista as pv


def main(argv):
    if not argv:
        return
    p = pv.Plotter(title="ra_diffusion_lite -- interactive view")
    ra = pv.read(argv[0])
    p.add_mesh(ra, color="lightsteelblue", opacity=0.30, smooth_shading=True)
    p.add_mesh(ra, style="wireframe", color="steelblue", line_width=1, opacity=0.25)
    if len(argv) > 1:
        p.add_mesh(pv.read(argv[1]), color="tan", smooth_shading=True)
    p.add_axes()
    # p.add_text("RA shell (blue, translucent)  +  NA cores (tan)", font_size=10)
    p.show()


if __name__ == "__main__":
    main(sys.argv[1:])
