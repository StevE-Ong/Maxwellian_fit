# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-purpose analysis tool: fit a hot-electron temperature, `dN/dE = C exp(-E/T)`,
to a particle energy spectrum (PIC output or experimental). Three files do the work —
`auto_maxwellian.py` (the fitter, NumPy only), `maxwellian_gui.py` (Tkinter front end),
`figformat.py` (the matplotlib rcParams used for publication figures) — plus
`Example_Maxwellian_fit.ipynb` as the worked example and `histogram_1.dat` as test data.

There is no package, no test suite, no linter config, and no build step. Scripts are run
from the repository directory.

## Commands

```console
# GUI (optionally with a file to load and fit on startup)
$ python maxwellian_gui.py histogram_1.dat

# re-run the example notebook in place, outputs and all
$ jupyter nbconvert --to notebook --execute --inplace Example_Maxwellian_fit.ipynb

# smoke-test the fitter after changing it; on histogram_1.dat the answer is
# T = 37.69 +/- 0.10 MeV over [173.9, 452.6] MeV, R^2 = 0.9994
$ python -c "import numpy as np, auto_maxwellian as am; \
c,e = np.loadtxt('histogram_1.dat', unpack=True, usecols=[0,1]); \
print(am.auto_maxwellian_fit(e, c).summary())"
```

On this machine the interpreter with matplotlib/NumPy/Jupyter is
`/home/ong/.anaconda3/envs/OpenPMD/bin/python`.

The GUI can be exercised without clicking: construct `SpectrumFitGUI(path)`, call
`fit_and_plot()`, inspect `app.status.get()` / `app.result`, `app.fig.savefig(...)`, then
`app.destroy()`. Needs a display (`DISPLAY=:1` here). Do not screenshot the root window
to check it — that captures the user's whole desktop.

## How the fit works

A PIC spectrum on a log plot has four regions: a steep cold bulk, a knee, the hot
exponential tail (the only genuinely straight part), and a sparse high-E floor where bins
hold one or two macroparticles. `auto_maxwellian_fit` scans every candidate window over
the merged spectrum and keeps the one spanning the widest energy range that survives all
the guards. Each guard exists because a specific failure mode was hit, and removing one
brings that failure back:

- **Empirical noise** (`_empirical_sigma`, running MAD about a running median). A weighted
  PIC histogram has no fixed count quantum — macroparticle weights vary continuously — so
  a bin's Poisson error cannot be recovered from its value. Do not reintroduce a
  quantum/Poisson noise model; it fails on every real spectrum.
- **Curvature significance** (`_curvature_significance`, quadratic term > 3σ rejects). A
  reduced-χ² ceiling alone swallows the knee and absorbs it as extra scatter.
- **Occupancy cut** (`_occupancy_cutoff`, drop everything above where local bin occupancy
  falls below 0.5). Merging the sparse floor produces densities set by the merge width,
  which reads as a long straight decline and can dominate the fit.
- **decades vs. observed** guard: a window's extrapolated falloff may not exceed the
  dynamic range actually present in it by more than one decade.

Do **not** add a "fit window holds < X% of the spectrum" rejection. The hot tail is always
a small number fraction (a few percent is healthy) because the cold bulk dominates the
count; `min_hot_fraction` exists but is off by default for that reason. Whether a snapshot
has a hot population at all needs an external energy scale, not the fitter.

`local_temperature()` gives the sliding-window `T_loc(E)`; its plateau is the tail, and it
is the fastest sanity check on a detected range.

## Figure conventions

The notebook and the GUI draw the same "autofit" figure, and it is the style the user
wants: raw histogram in grey, merged spectrum in navy (`#1f3b73`), amber (`#f5c518`)
shaded auto-detected range, red dashed fit line drawn **only over the fitted range**,
x-axis cut at `1.618 × Emax` (capped at the data end). Two rules, both from explicit user
corrections:

- **No `T_loc` curve** on the figure — it was drawn on a green twin axis and removed.
- **Temperature goes in the per-panel text annotation, never the legend.** A legend is
  drawn once per figure, so a temperature there is missing from every other panel of a
  multi-panel figure.

`plot_fit()` inside `auto_maxwellian.py` predates those decisions: it puts T in the legend
and draws `T_loc` by default. It is fine for quick interactive checks, but do not use it
as the model for a publication figure.

`figformat.figure_format()` returns rcParams with `text.usetex = True`, which needs a TeX
install. The GUI overrides it with a checkbox (off by default) and falls back to mathtext
if drawing raises; anything else that consumes `figure_format` should expect the same
failure mode.

## Data conventions

`histogram_1.dat` and the loaders are **counts in column 0, energy [MeV] in column 1** —
note the order, and that `np.loadtxt(..., unpack=True)` therefore yields `counts, energy`
while `auto_maxwellian_fit(energy, counts)` takes them the other way round. Energy must be
strictly increasing (the GUI sorts on load; `_centers_and_widths` raises otherwise).

The example notebook is committed with its outputs, and its last cell overwrites
`histogram.png`, which is the image the README embeds. That is intentional — but it means
executing the notebook is a tracked change.
