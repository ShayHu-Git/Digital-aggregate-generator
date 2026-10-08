"""
gui.py -- RA-NA Digital Aggregate Generator
===========================================
Single-window tkinter GUI over the ra_diffusion_lite pipeline.

Panels
  1 File                              reference RA / NA .ply, output folder
  2 Reference particle information     auto morphology (units in [ ])
  3 Reconstruction particle parameters k / alpha / l / EI / FI / size / ...
  4 3D view                            PyVista off-screen render + options
  5 Log info                           live pipeline log + report table

Conventions: EI = b/a  and  FI = c/b  (both <= 1).

Run:  python gui.py
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pyvista as pv

import pipeline
from _view import VTKView

HERE = Path(__file__).resolve().parent

FONT = ("Segoe UI", 9, "bold")
FONT_HDR = ("Segoe UI", 10, "bold")
FONT_VAL = ("Consolas", 9, "bold")

# panel 2 rows: (label with unit, key)
MORPH_ROWS = [
    ("a  (longest) [mm]", "a"),
    ("b  (intermediate) [mm]", "b"),
    ("c  (shortest) [mm]", "c"),
    ("Elongation  EI = b/a [-]", "EI"),
    ("Flatness  FI = c/b [-]", "FI"),
    ("Volume [mm3]", "volume"),
    ("Surface area [mm2]", "area"),
    ("Sphericity [-]", "sphericity"),
    ("NA cores  m [-]", "m"),
    ("Adhered mortar content [%]", "amc"),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digital Aggregate Generator")
        self.geometry("1280x820")
        self.minsize(1140, 720)

        self.q: queue.Queue = queue.Queue()
        self.ref_meshes = None
        self.analysis = None
        self.result = None
        self._analyzed_files = None          # (ra_path, na_path) of the current analysis
        self.busy = False

        self._style()
        self._build_left()
        self._build_right()
        self.after(80, self._drain)

    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", font=FONT)
        s.configure("TLabelframe.Label", font=FONT_HDR)
        s.configure("TButton", font=FONT)
        s.configure("Accent.TButton", font=("Segoe UI", 9, "bold"))
        s.configure("Val.TLabel", font=FONT_VAL)

    # ------------------------------------------------------------------ #
    def _build_left(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")

        # File ------------------------------------------------------- #
        f1 = ttk.LabelFrame(left, text=" File ", padding=8)
        f1.pack(fill="x", pady=(0, 6))
        self.v_ra = tk.StringVar()
        self.v_na = tk.StringVar()
        self.v_out = tk.StringVar(value=str(HERE / "out"))
        self._file_row(f1, "Reference RA (.ply)", self.v_ra, self._pick_ra, 0)
        self._file_row(f1, "Reference NA (.ply)", self.v_na, self._pick_na, 1)
        self._file_row(f1, "Output folder", self.v_out, self._pick_out, 2)
        ttk.Button(f1, text="Load & analyze reference", style="Accent.TButton",
                   command=self.on_analyze).grid(row=3, column=0, columnspan=3,
                                                 sticky="ew", pady=(6, 0))
        f1.columnconfigure(1, weight=1)

        # Reference particle information ------------------------- #
        f2 = ttk.LabelFrame(left, text=" Reference particle information ", padding=8)
        f2.pack(fill="x", pady=6)
        self.m_lbls = {}
        for i, (txt, key) in enumerate(MORPH_ROWS):
            ttk.Label(f2, text=txt).grid(row=i, column=0, sticky="w")
            lb = ttk.Label(f2, text="--", style="Val.TLabel")
            lb.grid(row=i, column=1, sticky="e")
            self.m_lbls[key] = lb
        f2.columnconfigure(1, weight=1)

        # Reconstruction particle parameters ------------------ #
        f3 = ttk.LabelFrame(left, text=" Reconstruction particle parameters ", padding=8)
        f3.pack(fill="x", pady=6)
        r = 0
        self.v_k = tk.IntVar(value=0)
        self.lbl_k = ttk.Label(f3, text="NA cores  k   [0, m] [-]")
        self.lbl_k.grid(row=r, column=0, sticky="w")
        self.sp_k = ttk.Spinbox(f3, from_=0, to=0, textvariable=self.v_k, width=8)
        self.sp_k.grid(row=r, column=1, sticky="e"); r += 1

        self.v_alpha = tk.DoubleVar(value=0.5)
        self._scale_row(f3, "Deviation  alpha  [0, 1] [-]", self.v_alpha, 0.0, 1.0, r); r += 1

        self.v_l = tk.IntVar(value=1)
        ttk.Label(f3, text="Particles to generate  l [-]").grid(row=r, column=0, sticky="w")
        ttk.Spinbox(f3, from_=1, to=999, textvariable=self.v_l, width=8).grid(
            row=r, column=1, sticky="e"); r += 1

        self.v_custom = tk.BooleanVar(value=False)
        ttk.Checkbutton(f3, text="Apply custom target morphology", variable=self.v_custom,
                        command=self._toggle_custom).grid(row=r, column=0, columnspan=2,
                                                          sticky="w", pady=(4, 0)); r += 1
        self.v_ei = tk.DoubleVar(value=0.7)
        self.v_fi = tk.DoubleVar(value=0.7)
        self.row_ei = self._scale_row(f3, "Target elongation  EI = b/a [-]",
                                      self.v_ei, 0.4, 1.0, r); r += 1
        self.row_fi = self._scale_row(f3, "Target flatness  FI = c/b [-]",
                                      self.v_fi, 0.4, 1.0, r); r += 1
        self.v_smin = tk.DoubleVar(value=0.0)
        self.v_smax = tk.DoubleVar(value=0.0)
        self.row_size = ttk.Frame(f3)
        self.row_size.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        ttk.Label(self.row_size, text="Size fraction  b [mm]  in [").pack(side="left")
        ttk.Entry(self.row_size, textvariable=self.v_smin, width=7).pack(side="left")
        ttk.Label(self.row_size, text=" , ").pack(side="left")
        ttk.Entry(self.row_size, textvariable=self.v_smax, width=7).pack(side="left")
        ttk.Label(self.row_size, text="]").pack(side="left")

        adv = ttk.Frame(f3)
        adv.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 0)); r += 1
        ttk.Label(adv, text="N [-]").pack(side="left")
        self.v_n = tk.IntVar(value=64)
        ttk.Combobox(adv, textvariable=self.v_n, values=[48, 64, 80, 96], width=4,
                     state="readonly").pack(side="left", padx=(2, 8))
        ttk.Label(adv, text="smooth [-]").pack(side="left")
        self.v_sig = tk.DoubleVar(value=0.6)
        ttk.Entry(adv, textvariable=self.v_sig, width=5).pack(side="left", padx=(2, 8))
        ttk.Label(adv, text="margin [mm]").pack(side="left")
        self.v_mrg = tk.DoubleVar(value=0.0)
        ttk.Entry(adv, textvariable=self.v_mrg, width=5).pack(side="left", padx=2)

        f3.columnconfigure(1, weight=1)
        self._toggle_custom()

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(4, 0))
        self.btn_gen = ttk.Button(btns, text="Generate", style="Accent.TButton",
                                  command=self.on_generate, state="disabled")
        self.btn_gen.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.btn_save = ttk.Button(btns, text="Save",
                                   command=self.on_save, state="disabled")
        self.btn_save.pack(side="left", expand=True, fill="x", padx=(3, 0))

    def _file_row(self, parent, label, var, cmd, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="...", width=3, command=cmd).grid(row=row, column=2)

    def _scale_row(self, parent, label, var, lo, hi, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        fr = ttk.Frame(parent)
        fr.grid(row=row, column=1, sticky="e")
        ttk.Entry(fr, textvariable=var, width=7).pack(side="right")
        ttk.Scale(fr, from_=lo, to=hi, variable=var, length=110,
                  command=lambda _v: var.set(round(float(var.get()), 3))).pack(
                      side="right", padx=(0, 4))
        return fr

    def _toggle_custom(self):
        st = "normal" if self.v_custom.get() else "disabled"
        for fr in (self.row_ei, self.row_fi, self.row_size):
            for w in fr.winfo_children():
                try:
                    w.configure(state=st)
                except tk.TclError:
                    pass

    # ------------------------------------------------------------------ #
    def _build_right(self):
        right = ttk.Frame(self, padding=8)
        right.pack(side="left", fill="both", expand=True)
        # deterministic 3 : 2 vertical split -- grid weights, NOT pack(expand),
        # so the 3D view can never grow past its share and cover the log
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)

        top = ttk.LabelFrame(right, text=" 3D view ", padding=6)
        top.grid(row=0, column=0, sticky="nsew")
        # particle selection lives inside VTKView: one "Show:" per panel once
        # two panels are open (after a generation).
        self.view = VTKView(top)
        self.view.pack(fill="both", expand=True)

        bot = ttk.LabelFrame(right, text=" Log info ", padding=6)
        bot.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.txt = tk.Text(bot, height=8, wrap="word", font=("Consolas", 9))
        sb = ttk.Scrollbar(bot, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(side="left", fill="both", expand=True)

    # ---- pickers ---------------------------------------------------- #
    def _pick_ra(self):
        p = filedialog.askopenfilename(filetypes=[("PLY", "*.ply"), ("All", "*.*")])
        if p:
            self.v_ra.set(p)
            g = str(Path(p).with_name(Path(p).stem + "_NA.ply"))
            if not self.v_na.get() and Path(g).exists():
                self.v_na.set(g)

    def _pick_na(self):
        p = filedialog.askopenfilename(filetypes=[("PLY", "*.ply"), ("All", "*.*")])
        if p:
            self.v_na.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.v_out.set(p)

    # ---- worker plumbing ---------------------------------------- #
    def log(self, msg):
        self.q.put(("log", str(msg)))

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.txt.insert("end", payload + "\n")
                    self.txt.see("end")
                elif kind == "call":
                    payload()
        except queue.Empty:
            pass
        self.after(80, self._drain)

    def _run_bg(self, fn):
        if self.busy:
            return
        self.busy = True
        self.btn_gen.configure(state="disabled")
        self.btn_save.configure(state="disabled")

        def wrap():
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                import traceback
                self.q.put(("log", "ERROR: " + str(e)))
                self.q.put(("log", traceback.format_exc()))
            finally:
                self.busy = False
                self.q.put(("call", self._after_bg))

        threading.Thread(target=wrap, daemon=True).start()

    def _after_bg(self):
        self.btn_gen.configure(state="normal" if self.analysis else "disabled")
        self.btn_save.configure(state="normal" if self.result else "disabled")

    # ---- 2 analyze --------------------------------------------- #
    def _reset_state(self):
        """Wipe everything tied to the previous reference / generation so a
        newly loaded reference starts the whole flow from scratch."""
        self.analysis = None
        self.ref_meshes = None
        self.result = None
        self._analyzed_files = None
        for lb in self.m_lbls.values():
            lb.configure(text="--")
        self.sp_k.configure(to=0)
        self.v_k.set(0)
        self.lbl_k.configure(text="NA cores  k   [0, m] [-]")
        self.btn_gen.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.view.clear()
        self.txt.delete("1.0", "end")

    def on_analyze(self):
        if self.busy:
            return
        ra, na = self.v_ra.get().strip(), self.v_na.get().strip()
        if not (ra and na and Path(ra).exists() and Path(na).exists()):
            messagebox.showerror("Missing file", "Pick both reference RA and NA .ply files.")
            return
        self._reset_state()          # discard any previous reference / particles

        def job():
            self.log(f"\n=== analyzing {Path(ra).name} / {Path(na).name} ===")
            a = pipeline.analyze_reference(ra, na)
            self.analysis = a
            self.ref_meshes = (pv.read(ra).clean().triangulate(),
                               pv.read(na).clean().triangulate())
            self.q.put(("call", lambda: self._fill_analysis(a)))
            r = a["ra"]
            self.log(f"a/b/c [mm] = {r['a']:.2f} / {r['b']:.2f} / {r['c']:.2f}")
            self.log(f"EI (b/a) = {r['EI']:.3f}   FI (c/b) = {r['FI']:.3f}   "
                     f"Volume [mm3] = {r['volume']:.1f}   Area [mm2] = {r['area']:.1f}")
            self.log(f"Sphericity = {r['sphericity']:.3f}   NA cores m = {a['m']}   "
                     f"Adhered mortar content = {a['amc'] * 100:.1f} %")
            self.log(f"NA core volumes [mm3] (large->small): "
                     f"{[round(v, 2) for v in a['core_volumes']]}")

        self._run_bg(job)

    def _fill_analysis(self, a):
        r = a["ra"]
        vals = dict(a=f"{r['a']:.2f}", b=f"{r['b']:.2f}", c=f"{r['c']:.2f}",
                    EI=f"{r['EI']:.3f}", FI=f"{r['FI']:.3f}",
                    volume=f"{r['volume']:.1f}", area=f"{r['area']:.1f}",
                    sphericity=f"{r['sphericity']:.3f}", m=str(a["m"]),
                    amc=f"{a['amc'] * 100:.1f}")
        for k, lb in self.m_lbls.items():
            lb.configure(text=vals.get(k, "--"))
        m = a["m"]
        self.sp_k.configure(to=m)
        self.v_k.set(m)
        self.lbl_k.configure(text=f"NA cores  k   [0, {m}] [-]")
        self.v_ei.set(round(r["EI"], 3))
        self.v_fi.set(round(r["FI"], 3))
        self.v_smin.set(round(r["b"], 2))
        self.v_smax.set(round(r["b"], 2))
        self._analyzed_files = (self.v_ra.get().strip(), self.v_na.get().strip())
        self.view.set_meshes(*self.ref_meshes)          # single panel until a generation
        self.btn_gen.configure(state="normal")

    # ---- 3 generate ------------------------------------------ #
    def on_generate(self):
        if self.busy or not self.analysis:
            return
        if (self.v_ra.get().strip(), self.v_na.get().strip()) != self._analyzed_files:
            messagebox.showwarning(
                "Reference changed",
                "The reference files differ from the analyzed ones.\n"
                "Click 'Load & analyze reference' first.")
            return
        try:
            p = pipeline.GenParams(
                ra_ply=self.v_ra.get().strip(), na_ply=self.v_na.get().strip(),
                out_dir=self.v_out.get().strip() or str(HERE / "out"),
                k=int(self.v_k.get()), alpha=float(self.v_alpha.get()),
                n_particles=int(self.v_l.get()), custom_shape=bool(self.v_custom.get()),
                target_EI=float(self.v_ei.get()), target_FI=float(self.v_fi.get()),
                size_min=float(self.v_smin.get()), size_max=float(self.v_smax.get()),
                N=int(self.v_n.get()), smooth_sigma=float(self.v_sig.get()),
                margin=float(self.v_mrg.get()),
            )
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Bad parameter", str(e))
            return

        def job():
            self.log(f"\n=== generating {p.n_particles} particle(s) "
                     f"(k={p.k}, alpha={p.alpha}) ===")
            res = pipeline.generate(p, progress=self.log)
            self.result = res
            self.q.put(("call", self._after_generate))
            self.log("\n" + res["report"])

        self._run_bg(job)

    def _after_generate(self):
        # open the two-panel view: Reference + every generated particle, each
        # panel with its own "Show:" selector.  It starts on Reference (left)
        # vs. Particle 1 (right) with both cameras aligned.
        mapping = {"Reference": self.ref_meshes}
        for i, (m_ra, m_na, _) in enumerate(self.result["particles"], start=1):
            mapping[f"Particle {i}"] = (m_ra, m_na)
        self.view.set_gallery(mapping)
        self.btn_save.configure(state="normal")

    # ---- 1 save ------------------------------------------------- #
    def on_save(self):
        if not self.result:
            return
        default_dir = self.v_out.get().strip() or str(HERE / "out")
        if not Path(default_dir).exists():
            default_dir = str(HERE)
        out = filedialog.askdirectory(initialdir=default_dir,
                                      title="Select folder to save generated particle(s)")
        if not out:
            return                                  # user cancelled -- save nothing
        self.v_out.set(out)                          # remember choice for next time
        stem = Path(self.v_ra.get()).stem
        try:
            paths = pipeline.save_particles(self.result, out, stem)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Save failed", str(e))
            return
        self.log(f"\nsaved {len(paths)} files to {out}:")
        for pth in paths:
            self.log("  " + Path(pth).name)


if __name__ == "__main__":
    App().mainloop()
