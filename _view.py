"""
_view.py
========
The GUI's "3D view" panel.

Rendering is done by PyVista itself (an off-screen VTK render), shown as an
image that refreshes on demand -- fully interactive mouse-drag embedded
PyVista needs Qt / pyvistaqt or the VTK-Tk widget DLL, neither of which is
present here.  Instead:

  * drag the image with the left mouse button to orbit
  * the arrow / +/- buttons orbit / zoom, "Reset" restores the start view
  * "Pop out" opens the real interactive PyVista window (_pvview.py)

Render options: display mode (surface / surface+edges / wireframe / points),
RA and NA colour, RA and NA opacity -- one shared set, applied identically to
every panel.

Two layouts:
  * set_meshes(ra, na)     -- one particle, one panel.
  * set_gallery(mapping)   -- two panels side by side.  `mapping` is an
                             ordered {name: (ra, na)} of every selectable
                             particle (Reference + each generated one).  Each
                             panel has its own "Show:" selector and its own
                             "Pop out" button, and each panel's camera orbits
                             / zooms independently (drag or wheel over that
                             half of the image; the nav buttons nudge both).
                             Both cameras start from the same pose and share a
                             common scale; "Reset" re-syncs them.
"""
from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import pyvista as pv
from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
PV_VIEWER = HERE / "_pvview.py"
TMP = HERE / "data" / "_gui_tmp"
TMP.mkdir(parents=True, exist_ok=True)

_MODES = ("surface", "surface + edges", "wireframe", "points")
_COLORS = ("steelblue", "tan", "lightgray", "salmon", "seagreen", "gold",
           "slategray", "white", "sandybrown")
_CAM0 = (35.0, -18.0, 1.0)          # default (azimuth, elevation, zoom)


class VTKView(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.ra = self.na = self._ra_r = self._na_r = None
        self._gallery: dict = {}                  # name -> (ra_full, na_full)
        self._dec: dict = {}                      # name -> (ra_dec, na_dec)  (lazy)
        self._compare = False
        self.sel = ["", ""]                       # selected name per panel (compare mode)
        self.cam = [list(_CAM0), list(_CAM0)]     # [az, el, zoom] per panel
        self._last = None
        self._drag_side = 0
        self._pending = False
        self._imgtk = None

        self._build_controls()
        # The image Label must NOT drive the layout: a tk.Label sized to its
        # image would keep growing this frame every render and swallow the
        # panel below it.  Put it in a size-locked holder (pack_propagate off)
        # and `place` it -- placed widgets contribute nothing to the parent's
        # requested size, so the 3D view stays inside its allotted share.
        holder = ttk.Frame(self, width=520, height=360)
        holder.pack(side="top", fill="both", expand=True)
        holder.pack_propagate(False)
        self._holder = holder
        self.canvas = tk.Label(holder, bg="white", cursor="fleur")
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)
        holder.bind("<Configure>", lambda _e: self._schedule())
        for seq, fn in (("<ButtonPress-1>", self._press), ("<B1-Motion>", self._drag),
                        ("<MouseWheel>", self._wheel)):
            self.canvas.bind(seq, fn)
        self._apply_mode()

    # ---- controls -------------------------------------------------- #
    def _build_controls(self):
        # per-panel selectors + pop-out -- shown only in the two-panel layout
        self._topbar = ttk.Frame(self)
        self._sel_l, self._sel_r = tk.StringVar(), tk.StringVar()
        for side, var, anchor in ((0, self._sel_l, "left"), (1, self._sel_r, "right")):
            grp = ttk.Frame(self._topbar)
            grp.pack(side=anchor)
            ttk.Label(grp, text=("Left  Show:" if side == 0 else "Right  Show:")).pack(side="left")
            cbo = ttk.Combobox(grp, textvariable=var, width=14, state="readonly")
            cbo.pack(side="left", padx=3)
            cbo.bind("<<ComboboxSelected>>", lambda _e, s=side: self._on_select(s))
            ttk.Button(grp, text="Pop out", width=8,
                       command=lambda s=side: self._popout(s)).pack(side="left", padx=(0, 6))
            setattr(self, f"_cbo_{side}", cbo)

        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x", pady=(3, 0))
        self.v_mode = tk.StringVar(value=_MODES[0])
        ttk.Label(bar, text="Mode").pack(side="left")
        self._cbo(bar, self.v_mode, _MODES, 13)
        ttk.Label(bar, text=" RA").pack(side="left")
        self.v_rac = tk.StringVar(value="steelblue")
        self._cbo(bar, self.v_rac, _COLORS, 10)
        self.v_rao = tk.DoubleVar(value=0.35)
        self._sld(bar, self.v_rao)
        ttk.Label(bar, text=" NA").pack(side="left")
        self.v_nac = tk.StringVar(value="tan")
        self._cbo(bar, self.v_nac, _COLORS, 10)
        self.v_nao = tk.DoubleVar(value=1.0)
        self._sld(bar, self.v_nao)

        nav = ttk.Frame(self)
        nav.pack(side="bottom", fill="x")
        for txt, fn in (("◄", lambda: self._orbit(-20, 0)),
                        ("►", lambda: self._orbit(20, 0)),
                        ("▲", lambda: self._orbit(0, 15)),
                        ("▼", lambda: self._orbit(0, -15)),
                        ("+", lambda: self._zoomby(1.25)),
                        ("−", lambda: self._zoomby(0.8)),
                        ("Reset", self.reset_camera)):
            ttk.Button(nav, text=txt, width=4 if len(txt) < 3 else 7,
                       command=fn).pack(side="left", padx=1)
        self._nav_popout = ttk.Button(nav, text="Pop out (PyVista)",
                                      command=lambda: self._popout(0))
        self._nav_popout.pack(side="right")

    def _cbo(self, bar, var, values, w):
        c = ttk.Combobox(bar, textvariable=var, values=list(values), width=w, state="readonly")
        c.pack(side="left", padx=2)
        c.bind("<<ComboboxSelected>>", lambda _e: self._schedule())

    def _sld(self, bar, var):
        ttk.Scale(bar, from_=0.05, to=1.0, variable=var, length=70,
                  command=lambda _v: self._schedule()).pack(side="left", padx=(2, 6))

    def _apply_mode(self):
        if self._compare:
            self._topbar.pack(side="top", fill="x", before=self._holder, pady=(0, 3))
            self._nav_popout.pack_forget()
        else:
            self._topbar.pack_forget()
            self._nav_popout.pack(side="right")

    # ---- mesh registry ------------------------------------------ #
    @staticmethod
    def _light(mesh, max_faces=9000):
        if mesh is None:
            return None
        m = mesh.triangulate()
        if m.n_cells > max_faces:
            try:
                m = m.decimate(1.0 - max_faces / m.n_cells)
            except Exception:
                pass
        return m

    def _resolve(self, name):
        """(ra_dec, na_dec) for a gallery entry, decimated on first use."""
        if name not in self._dec:
            ra, na = self._gallery.get(name, (None, None))
            if na is not None and not getattr(na, "n_points", 0):
                na = None
            self._dec[name] = (self._light(ra), self._light(na))
        return self._dec[name]

    def clear(self):
        """Drop every mesh and blank the canvas -- used when a new reference is
        loaded so nothing from the previous reference / generation lingers."""
        self._compare = False
        self._gallery = {}
        self._dec = {}
        self.sel = ["", ""]
        self.ra = self.na = self._ra_r = self._na_r = None
        self.cam = [list(_CAM0), list(_CAM0)]
        self._apply_mode()
        self._imgtk = None
        self.canvas.configure(image="")

    def set_meshes(self, ra, na):
        """One particle, one panel."""
        self._compare = False
        self._gallery = {"model": (ra, na)}
        self._dec = {}
        self.sel = ["", ""]
        self.ra = ra
        self.na = na if (na is not None and getattr(na, "n_points", 0)) else None
        self._ra_r, self._na_r = self._resolve("model")
        self._apply_mode()
        self.reset_camera()

    def set_gallery(self, mapping: dict):
        """Two panels.  `mapping` = ordered {name: (ra, na)} of every particle
        the user may pick from (Reference + each generated particle)."""
        self._gallery = dict(mapping)
        self._dec = {}
        names = list(self._gallery) or [""]
        self._compare = True
        self.sel = [names[0], names[1] if len(names) > 1 else names[0]]
        self._sel_l.set(self.sel[0])
        self._sel_r.set(self.sel[1])
        self._cbo_0.configure(values=names)
        self._cbo_1.configure(values=names)
        self._apply_mode()
        self.reset_camera()

    def _on_select(self, side):
        self.sel[side] = (self._sel_l if side == 0 else self._sel_r).get()
        self._schedule()

    # ---- camera -------------------------------------------------- #
    def reset_camera(self):
        self.cam = [list(_CAM0), list(_CAM0)]     # both panels start identical
        self._schedule()

    def _sides(self):
        return (0, 1) if self._compare else (0,)

    def _panel_at(self, x):
        if not self._compare:
            return 0
        return 0 if x < max(1, self.canvas.winfo_width()) // 2 else 1

    def _press(self, e):
        self._last = (e.x, e.y)
        self._drag_side = self._panel_at(e.x)

    def _drag(self, e):
        if self._last is None:
            return
        dx, dy = e.x - self._last[0], e.y - self._last[1]
        self._last = (e.x, e.y)
        c = self.cam[self._drag_side]              # the panel the drag started in
        c[0] -= dx * 0.4
        c[1] = max(-89, min(89, c[1] + dy * 0.4))
        self._schedule()

    def _wheel(self, e):
        c = self.cam[self._panel_at(e.x)]
        c[2] = max(0.2, min(6.0, c[2] * (1.15 if e.delta > 0 else 0.87)))
        self._schedule()

    def _orbit(self, daz, dele):
        for i in self._sides():                    # nav buttons nudge every panel
            c = self.cam[i]
            c[0] += daz
            c[1] = max(-89, min(89, c[1] + dele))
        self._schedule()

    def _zoomby(self, f):
        for i in self._sides():
            self.cam[i][2] = max(0.2, min(6.0, self.cam[i][2] * f))
        self._schedule()

    # ---- pop out ---------------------------------------------- #
    def _popout(self, side):
        pair = self._gallery.get(self.sel[side]) if self._compare else self._gallery.get("model")
        if not pair or pair[0] is None:
            return
        ra, na = pair
        ra_p = str(TMP / f"view_{side}_RA.ply")
        ra.save(ra_p)
        args = [sys.executable, str(PV_VIEWER), ra_p]
        if na is not None and getattr(na, "n_points", 0):
            na_p = str(TMP / f"view_{side}_NA.ply")
            na.save(na_p)
            args.append(na_p)
        subprocess.Popen(args, cwd=str(HERE))

    # ---- rendering ------------------------------------------- #
    def _schedule(self):
        if self._pending:
            return
        self._pending = True
        self.after(40, self._render)

    def _add_particle(self, pl, ra_r, na_r, mode, style):
        """One RA(+NA) pair with the current render options -- shared by the
        single view and by each panel of the two-panel view, so the settings
        the panels use are guaranteed identical."""
        se = (mode == "surface + edges")
        if ra_r is not None:
            pl.add_mesh(ra_r, color=self.v_rac.get(), opacity=float(self.v_rao.get()),
                        style=style, show_edges=se, edge_color="#333333",
                        line_width=1, point_size=3, smooth_shading=True)
        if na_r is not None and getattr(na_r, "n_points", 0):
            pl.add_mesh(na_r, color=self.v_nac.get(), opacity=float(self.v_nao.get()),
                        style=style, show_edges=se, edge_color="#333333",
                        line_width=1, point_size=3, smooth_shading=True)

    @staticmethod
    def _centre_ext(ra_r):
        if ra_r is None:
            return (0.0, 0.0, 0.0), 1.0
        b = np.asarray(ra_r.bounds).reshape(3, 2)
        return tuple((b[:, 0] + b[:, 1]) / 2.0), (float((b[:, 1] - b[:, 0]).max()) or 1.0)

    def _compose_single(self, w, h, mode, style):
        pl = pv.Plotter(off_screen=True, window_size=(w, h))
        pl.set_background("white", top="#e6edf5")
        self._add_particle(pl, self._ra_r, self._na_r, mode, style)
        pl.add_axes(viewport=(0, 0, 0.25, 0.3))
        pl.view_isometric()
        az, el, zoom = self.cam[0]
        pl.camera.azimuth = az
        pl.camera.elevation = el
        pl.camera.zoom(zoom)
        img = pl.screenshot(return_img=True)
        pl.close()
        return img

    def _compose_pair(self, w, h, mode, style):
        metas = [self._resolve(n) for n in self.sel]
        # one shared reference extent -> equal zoom gives equal apparent size
        ext_common = max(self._centre_ext(ra)[1] for ra, _ in metas)
        pl = pv.Plotter(off_screen=True, window_size=(w, h), shape=(1, 2),
                        border_color="#bcbcbc")
        pl.set_background("white", top="#e6edf5")
        for col in (0, 1):
            ra_r, na_r = metas[col]
            cen, _ = self._centre_ext(ra_r)
            az, el, zoom = self.cam[col]
            pl.subplot(0, col)
            self._add_particle(pl, ra_r, na_r, mode, style)
            pl.add_text(f"{['Left', 'Right'][col]}: {self.sel[col]}",
                        font_size=9, color="#333333")
            pl.add_axes()
            pl.enable_parallel_projection()
            pl.camera_position = "iso"
            pl.camera.azimuth = az
            pl.camera.elevation = el
            pl.camera.focal_point = cen
            pl.camera.parallel_scale = (0.60 / max(zoom, 1e-3)) * ext_common
        # deliberately NO link_views(): each panel's camera moves independently
        img = pl.screenshot(return_img=True)
        pl.close()
        return img

    def _render(self):
        self._pending = False
        w = min(1600, max(120, self._holder.winfo_width()))
        h = min(1000, max(120, self._holder.winfo_height()))
        mode = self.v_mode.get()
        style = {"surface": "surface", "surface + edges": "surface",
                 "wireframe": "wireframe", "points": "points"}[mode]
        try:
            if self._compare and self._gallery:
                img = self._compose_pair(w, h, mode, style)
            elif self._ra_r is not None:
                img = self._compose_single(w, h, mode, style)
            else:
                return
        except Exception as ex:  # noqa: BLE001
            print("render error:", ex)
            return
        self._imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.configure(image=self._imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("980x600")
    v = VTKView(root)
    v.pack(fill="both", expand=True)
    v.set_gallery({
        "Reference": (pv.Sphere(radius=1.0), pv.Sphere(radius=0.55)),
        "Particle 1": (pv.Sphere(radius=1.15), pv.Sphere(radius=0.5)),
        "Particle 2": (pv.Cube(), pv.Sphere(radius=0.4)),
    })
    root.after(2500, root.destroy)
    root.mainloop()
    print("VTKView (offscreen) OK")
