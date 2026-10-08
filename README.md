# Digital aggregate generator

A lightweight, CPU-only, training-free generator of 3D **recycled-aggregate
(RA) particles** for meso-scale modelling of recycled aggregate concrete.

Starting from one X-CT scanned reference particle, it produces any number of
new particles that inherit the reference's character without copying it. Each
particle is a pair of watertight meshes:

- `*_RA.ply` — the outer RA surface (natural aggregate + adhered old mortar)
- `*_NA.ply` — the natural-aggregate (NA) core(s) inside it

![Mid cross-sections of three references and four generated variants each](figs/montage_slices.png)

*Grey = inside the RA shell, red = inside an NA core. Left column: X-CT
reference; other columns: generated variants (`alpha = 0.5`).*

> **Note on scope.** The generator follows the single-example idea of
> SingleShapeGen / Sin3DM (resynthesise from one exemplar), but the generative
> core is a random smooth warp, **not a trained neural network**. A diffusion
> or GAN backbone can replace `synth.generate()` without touching the rest of
> the pipeline.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/ra-particle-generator.git
cd ra-particle-generator
pip install -r requirements.txt
```

`tkinter` (used by the GUI) ships with the standard Python installers on
Windows and macOS; on Debian/Ubuntu install it with `sudo apt install python3-tk`.

## Quick start

```bash
python gui.py              # interactive tool
python run.py              # validation run on the three example particles (~15 s)
python run.py --quick      # smoke test: A8 only, 2 seeds, N=48
python viz.py              # figures in out/ (run after run.py)
```

Or from Python:

```python
import pipeline

p = pipeline.GenParams(
    ra_ply="examples/ref/A2.ply",
    na_ply="examples/ref/A2_NA.ply",
    n_particles=5,
    alpha=0.5,
)
result = pipeline.generate(p, progress=print)
pipeline.save_particles(result, "out/demo", stem="A2")
```

## How it works

### Thickness-field representation

Each reference pair is voxelised to an `(N, N, N, 2)` grid on an anisotropic
world grid:

```
channel 0 :  d_RA(x)              signed distance to the RA surface
channel 1 :  T(x) = d_NA - d_RA   adhered-mortar thickness, clipped to >= 0
```

The NA surface is always rebuilt as the zero set of
`d_NA = d_RA + clip(T, 0, inf) + margin`. Because `T >= 0` and `margin >= 0`,
`d_NA >= d_RA` everywhere, so the NA core cannot leave the shell. This holds
by construction: there is no penalty term and no post-hoc repair. Where
`T = 0` the two surfaces coincide, which represents natural aggregate exposed
at the particle surface (partial mortar wrapping).

### Single-field warp

`synth.generate()` applies **one** smooth random displacement field to both
channels. Since `d_RA` and `T` move together, the thin mortar veins that
separate neighbouring NA cores stay registered and the cores do not merge.
`alpha` in `[0, 1]` scales the displacement: `0` returns the reference
unchanged, `0.5` is the default, values towards `1` give stronger variation.

### Pipeline

| File | Role |
|---|---|
| `represent.py` | voxelise a reference pair into the `[d_RA, T]` grid; keep the `k` largest NA cores |
| `synth.py` | generative core: `generate(ref_grid, seed, alpha) -> grid` |
| `reconstruct.py` | grid to watertight `mesh_RA`, `mesh_NA` (marching cubes, nesting-preserving smoothing) |
| `postprocess.py` | drop NA fragments below 0.3 mm³; volume matching; rescale to a target shape |
| `morphology.py` | PCA oriented bounding box, `EI`, `FI`, volume, area, sphericity |
| `metrics.py` | volumes, adhered-mortar content, containment, exposed fraction, diversity |
| `pipeline.py` | `analyze_reference()` and `generate(GenParams)`: the whole chain in one call |
| `gui.py`, `_view.py`, `_pvview.py` | tkinter GUI with a PyVista 3D view |
| `run.py` | batch validation over the example particles, writes `out/report.md` |
| `viz.py`, `paper_figures.py` | figures (`paper_figures.py` needs `scienceplots`) |

## Parameters

| Parameter | Meaning | Default |
|---|---|---|
| `alpha` | deviation from the reference, `[0, 1]` | `0.5` |
| `k` | number of NA cores kept (the `k` largest); `0` gives a mortar-only particle | all |
| `n_particles` | particles to generate | `1` |
| `custom_shape` | rescale each particle to a target morphology | `False` |
| `target_EI`, `target_FI` | target elongation `EI = b/a` and flatness `FI = c/b`, both in `(0, 1]` | reference values |
| `size_min`, `size_max` | band [mm] for the intermediate dimension `b` | reference `b` |
| `N` | voxel grid resolution | `64` |
| `smooth_sigma` | Gaussian smoothing of the fields [voxels] | `0.6` |
| `margin` | forced mortar gap between NA and RA surfaces [mm]; `0` allows exposed NA | `0.0` |

`a >= b >= c` are the dimensions of the PCA oriented bounding box. The target
rescale is one affine map applied to the RA shell and every NA core together,
so containment is preserved.

## GUI

`python gui.py` opens a single window with five panels:

1. **File** — reference RA and NA `.ply` (the NA file is guessed as
   `<ra_stem>_NA.ply`) and the output folder.
2. **Reference particle information** — dimensions, `EI`, `FI`, volume,
   surface area, sphericity, number of NA cores, adhered-mortar content.
3. **Reconstruction particle parameters** — `k`, `alpha`, number of particles,
   optional target morphology, advanced settings; **Generate**, then **Save**.
4. **3D view** — off-screen PyVista render (drag to orbit, wheel to zoom);
   **Pop out** opens a fully interactive PyVista window.
5. **Log info** — pipeline log and the generation report.

## Validation on the example particles

`python run.py` (3 references × 4 seeds, `N = 64`, `alpha = 0.5`) gives:

| Property | Result |
|---|---|
| NA outside RA, sampled on the generated `d_RA` field | 0.00 % on all 12 particles |
| Diversity: mean pairwise IoU of the RA shells across seeds | 0.64 for every reference |
| NA core count, generated / reference | A2 3/3, A7 1/1, A8 9–10/10 |
| Adhered-mortar content, generated vs reference | A2 0.30 / 0.30, A7 0.20–0.23 / 0.21, A8 0.72 / 0.70 |
| Exposed-NA surface fraction, generated vs reference | A2 0.50 / 0.47, A7 0.74 / 0.72, A8 0.32 / 0.34 |
| Time per particle | about 0.5 s |

The full table is written to `out/report.md`.

### Known limitations

- **Thin mortar veins.** In particles with many small cores (A8), the warp can
  blur the thinnest veins, so one core may merge with a neighbour.
- **Ray-cast containment check.** The independent `select_enclosed_points`
  check reports 1–2 % of NA vertices outside the shell for A2 and A8, and
  18–20 % for A7. These vertices lie where the NA surface coincides with the
  RA surface (exposed aggregate), where a ray-cast inside test is unreliable.
  Use `--margin` if a strict gap is required downstream.
- **Volume drift.** The warp changes the RA volume by up to about 17 % before
  correction; `run.py` rescales each particle back to the reference volume.

## Example data

`examples/ref/` holds three X-CT reference particles (`A2`, `A7`, `A8`), each
as an RA surface mesh and a matching `_NA` mesh. Any pair of closed `.ply`
meshes in the same coordinate system and in millimetres can be used instead.

## Citation

If you use this code in your research, please cite this repository.

## License

MIT — see [LICENSE](LICENSE).
