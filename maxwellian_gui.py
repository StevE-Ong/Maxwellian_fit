"""Desktop GUI for the automatic Maxwellian (hot-electron temperature) fit.

Load a spectrum file, let `auto_maxwellian` find the exponential range on its
own, view the plot, tune the figure, and save it.

    $ python maxwellian_gui.py [spectrum.dat]

Only the standard library and matplotlib are needed on top of the repository
itself (Tkinter ships with CPython; `auto_maxwellian` needs just NumPy).
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib as mpl
import numpy as np

mpl.use("TkAgg")

# so the app can be launched from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure  # noqa: E402

from auto_maxwellian import auto_maxwellian_fit  # noqa: E402

try:
    import figformat  # noqa: E402
except ImportError:  # keep the GUI usable without the repo's style module
    figformat = None

SCREEN_DPI = 100  # on-screen only; the save dpi is a user setting

RAW_COLOR = "0.75"
MERGED_COLOR = "#1f3b73"
RANGE_COLOR = "#f5c518"
FIT_COLOR = "red"


class SpectrumFitGUI(tk.Tk):
    def __init__(self, initial_file=None):
        super().__init__()
        self.title("Maxwellian fit")
        self.geometry("1180x720")

        self.energy = None
        self.counts = None
        self.result = None

        self._make_vars()
        self._build_layout()

        if initial_file:
            self.path.set(os.path.abspath(initial_file))
            self.load_file()

    # ------------------------------------------------------------------ state

    def _make_vars(self):
        v = tk.StringVar
        # data
        self.path = v(value="")
        self.col_counts = tk.IntVar(value=0)
        self.col_energy = tk.IntVar(value=1)
        self.skiprows = tk.IntVar(value=0)
        self.delimiter = v(value="")  # blank = any whitespace

        # fit (see auto_maxwellian.auto_maxwellian_fit for the meaning of each)
        self.n_target = tk.IntVar(value=150)
        self.min_decades = tk.DoubleVar(value=1.0)
        self.max_chi2 = tk.DoubleVar(value=3.0)
        self.max_curvature = tk.DoubleVar(value=3.0)
        self.min_bins = tk.IntVar(value=8)
        self.sigma_floor = tk.DoubleVar(value=0.05)
        self.min_occupancy = tk.DoubleVar(value=0.5)
        self.emin_floor = v(value="")
        self.emax_ceil = v(value="")

        # figure
        self.fig_width = tk.DoubleVar(value=3.4)
        self.fig_height = tk.DoubleVar(value=2.0)
        self.fontsize = tk.IntVar(value=7)
        self.save_dpi = tk.IntVar(value=300)
        self.usetex = tk.BooleanVar(value=False)
        self.title_txt = v(value="")
        self.xlabel = v(value="Energy (MeV)")
        self.ylabel = v(value=r"$dN/dE~(\mathrm{arb.\ units})$")
        self.xscale = v(value="linear")
        self.yscale = v(value="log")
        self.xmin = v(value="0")
        self.xmax = v(value="")  # blank = 1.618 x fitted Emax, capped at data
        self.ymin = v(value="")
        self.ymax = v(value="")

        # curves
        self.show_raw = tk.BooleanVar(value=True)
        self.show_merged = tk.BooleanVar(value=True)
        self.show_range = tk.BooleanVar(value=True)
        self.show_fit = tk.BooleanVar(value=True)
        self.show_annot = tk.BooleanVar(value=True)
        self.show_legend = tk.BooleanVar(value=True)
        self.minor_ticks = tk.BooleanVar(value=True)

        self.status = v(value="Select a spectrum file to begin.")

    # ----------------------------------------------------------------- layout

    def _build_layout(self):
        left = ttk.Frame(self, padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self, padding=(0, 6, 6, 6))
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(left, width=330)
        nb.pack(fill=tk.BOTH, expand=True)
        data_tab = ttk.Frame(nb, padding=6)
        fig_tab = ttk.Frame(nb, padding=6)
        nb.add(data_tab, text="Data & fit")
        nb.add(fig_tab, text="Figure")

        self._build_data_tab(data_tab)
        self._build_figure_tab(fig_tab)

        buttons = ttk.Frame(left, padding=(0, 6, 0, 0))
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Fit && plot", command=self.fit_and_plot).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        ttk.Button(buttons, text="Save figure...", command=self.save_figure).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        ttk.Button(buttons, text="Save report...", command=self.save_report).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Label(left, text="Fit result").pack(anchor="w", pady=(8, 0))
        self.report = tk.Text(left, height=10, width=42, wrap="none",
                              font=("TkFixedFont", 9))
        self.report.pack(fill=tk.X)
        self.report.configure(state=tk.DISABLED)

        # plot area
        self.fig = Figure(figsize=(self.fig_width.get(), self.fig_height.get()),
                          dpi=SCREEN_DPI)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN,
                  anchor="w", padding=3).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_data_tab(self, parent):
        f = ttk.LabelFrame(parent, text="Spectrum file", padding=6)
        f.pack(fill=tk.X)
        ttk.Entry(f, textvariable=self.path, width=34).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Button(f, text="Browse...", command=self.browse).grid(
            row=1, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(f, text="Reload", command=self.load_file).grid(
            row=1, column=1, sticky="ew")
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        c = ttk.LabelFrame(parent, text="Columns (0-based)", padding=6)
        c.pack(fill=tk.X, pady=(6, 0))
        self._row(c, 0, "counts / dN/dE column", self.col_counts, width=8)
        self._row(c, 1, "energy column [MeV]", self.col_energy, width=8)
        self._row(c, 2, "header rows to skip", self.skiprows, width=8)
        self._row(c, 3, "delimiter (blank = spaces)", self.delimiter, width=8)

        g = ttk.LabelFrame(parent, text="Auto-range detection", padding=6)
        g.pack(fill=tk.X, pady=(6, 0))
        self._row(g, 0, "n_target (bins after merge)", self.n_target)
        self._row(g, 1, "min_decades", self.min_decades)
        self._row(g, 2, "max_chi2", self.max_chi2)
        self._row(g, 3, "max_curvature [sigma]", self.max_curvature)
        self._row(g, 4, "min_bins", self.min_bins)
        self._row(g, 5, "sigma_floor", self.sigma_floor)
        self._row(g, 6, "min_occupancy", self.min_occupancy)
        self._row(g, 7, "search Emin floor [MeV]", self.emin_floor)
        self._row(g, 8, "search Emax ceiling [MeV]", self.emax_ceil)
        ttk.Label(g, text="Blank floor/ceiling = search the whole spectrum.",
                  wraplength=300, foreground="#555").grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_figure_tab(self, parent):
        s = ttk.LabelFrame(parent, text="Size and fonts", padding=6)
        s.pack(fill=tk.X)
        self._row(s, 0, "width [in]", self.fig_width)
        self._row(s, 1, "height [in]", self.fig_height)
        self._row(s, 2, "font size [pt]", self.fontsize)
        self._row(s, 3, "save dpi", self.save_dpi)
        ttk.Checkbutton(s, text="LaTeX text rendering (needs a TeX install)",
                        variable=self.usetex).grid(row=4, column=0, columnspan=2,
                                                   sticky="w", pady=(4, 0))

        a = ttk.LabelFrame(parent, text="Axes", padding=6)
        a.pack(fill=tk.X, pady=(6, 0))
        self._row(a, 0, "title", self.title_txt, width=18)
        self._row(a, 1, "x label", self.xlabel, width=18)
        self._row(a, 2, "y label", self.ylabel, width=18)
        ttk.Label(a, text="x scale").grid(row=3, column=0, sticky="w")
        ttk.Combobox(a, textvariable=self.xscale, values=("linear", "log"),
                     state="readonly", width=8).grid(row=3, column=1, sticky="e")
        ttk.Label(a, text="y scale").grid(row=4, column=0, sticky="w")
        ttk.Combobox(a, textvariable=self.yscale, values=("log", "linear"),
                     state="readonly", width=8).grid(row=4, column=1, sticky="e")
        self._row(a, 5, "x min", self.xmin)
        self._row(a, 6, "x max", self.xmax)
        self._row(a, 7, "y min", self.ymin)
        self._row(a, 8, "y max", self.ymax)
        ttk.Label(a, text="Blank limit = automatic (x max defaults to "
                          "1.618 x the fitted Emax).",
                  wraplength=300, foreground="#555").grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(4, 0))

        e = ttk.LabelFrame(parent, text="Elements", padding=6)
        e.pack(fill=tk.X, pady=(6, 0))
        for text, var in (("raw spectrum", self.show_raw),
                          ("merged spectrum", self.show_merged),
                          ("shaded fit range", self.show_range),
                          ("fit line", self.show_fit),
                          ("temperature annotation", self.show_annot),
                          ("legend", self.show_legend),
                          ("minor ticks", self.minor_ticks)):
            ttk.Checkbutton(e, text=text, variable=var).pack(anchor="w")

        ttk.Button(parent, text="Redraw", command=self.redraw).pack(
            fill=tk.X, pady=(8, 0))

    @staticmethod
    def _row(parent, row, label, var, width=10):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        ttk.Entry(parent, textvariable=var, width=width, justify="right").grid(
            row=row, column=1, sticky="e", pady=1)
        parent.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------- data

    def browse(self):
        name = filedialog.askopenfilename(
            title="Open spectrum",
            filetypes=[("Spectrum data", "*.dat *.txt *.csv"), ("All files", "*")],
            initialdir=os.path.dirname(self.path.get()) or os.getcwd(),
        )
        if name:
            self.path.set(name)
            self.load_file()

    def load_file(self):
        name = self.path.get().strip()
        if not name:
            return False
        try:
            delim = self.delimiter.get().strip() or None
            counts, energy = np.loadtxt(
                name,
                unpack=True,
                usecols=[self.col_counts.get(), self.col_energy.get()],
                skiprows=self.skiprows.get(),
                delimiter=delim,
                dtype=float,
            )
        except Exception as exc:
            messagebox.showerror("Could not read file", f"{name}\n\n{exc}")
            self._set_status("Load failed.")
            return False

        order = np.argsort(energy)  # the fitter needs increasing energy
        self.energy, self.counts = energy[order], counts[order]
        self.result = None
        self._set_status(
            f"Loaded {os.path.basename(name)}: {self.energy.size} bins, "
            f"E = {self.energy.min():.3g}-{self.energy.max():.3g} MeV")
        return True

    # -------------------------------------------------------------------- fit

    def _optional(self, var):
        text = var.get().strip()
        return float(text) if text else None

    def fit_and_plot(self):
        if self.energy is None and not self.load_file():
            return
        if self.energy is None:
            messagebox.showinfo("No data", "Choose a spectrum file first.")
            return
        try:
            kw = dict(
                n_target=self.n_target.get(),
                min_decades=self.min_decades.get(),
                max_chi2=self.max_chi2.get(),
                max_curvature=self.max_curvature.get(),
                min_bins=self.min_bins.get(),
                sigma_floor=self.sigma_floor.get(),
                min_occupancy=self.min_occupancy.get(),
                Emin_floor=self._optional(self.emin_floor),
                Emax_ceil=self._optional(self.emax_ceil),
            )
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Bad fit setting", str(exc))
            return

        self._set_status("Fitting...")
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            self.result = auto_maxwellian_fit(self.energy, self.counts, **kw)
        except Exception as exc:
            self.configure(cursor="")
            messagebox.showerror("Fit error", str(exc))
            self._set_status("Fit error.")
            return
        finally:
            self.configure(cursor="")

        self._show_report()
        self.redraw()
        r = self.result
        self._set_status(
            f"T = {r.T:.2f} +/- {r.T_err:.2f} MeV over "
            f"[{r.Emin:.1f}, {r.Emax:.1f}] MeV, R2 = {r.r_squared:.4f}"
            if r.ok else f"Fit failed: {r.message}")

    def _show_report(self):
        r = self.result
        text = r.summary() if r is not None else ""
        if r is not None and r.ok:
            text += (f"\nwindow holds {r.window_fraction:.2%} of the particles"
                     f"\nmerge factor = x{r.merge_factor}")
        self.report.configure(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert("1.0", text)
        self.report.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------- plot

    def _apply_style(self):
        if figformat is not None:
            _, _, params = figformat.figure_format(
                fig_width=self.fig_width.get(), fig_height=self.fig_height.get())
        else:
            params = {"font.family": "serif", "legend.frameon": False,
                      "xtick.direction": "in", "ytick.direction": "in",
                      "xtick.top": True, "ytick.right": True}
        params["text.usetex"] = bool(self.usetex.get())
        size = self.fontsize.get()
        for key in ("axes.labelsize", "axes.titlesize", "font.size",
                    "legend.fontsize", "xtick.labelsize", "ytick.labelsize"):
            params[key] = size
        mpl.rcParams.update(params)

    def redraw(self):
        if self.energy is None:
            return
        try:
            self._draw()
        except RuntimeError as exc:
            # almost always a missing LaTeX installation
            if self.usetex.get():
                self.usetex.set(False)
                self._set_status(f"LaTeX rendering failed, using mathtext ({exc})")
                self._draw()
            else:
                raise

    def _draw(self):
        self._apply_style()
        self.fig.clf()
        self.fig.set_size_inches(self.fig_width.get(), self.fig_height.get())
        self.fig.patch.set_facecolor("white")
        ax = self.fig.add_subplot(111)

        E, c = self.energy, self.counts
        m = c > 0
        r = self.result

        if self.show_raw.get():
            ax.plot(E[m], c[m], color=RAW_COLOR, label="raw")
        if r is not None and r.ok:
            if self.show_merged.get():
                ax.plot(r.E_rebin, r.dNdE_rebin, color=MERGED_COLOR,
                        label=rf"merged ($\times${r.merge_factor})")
            if self.show_range.get():
                ax.axvspan(r.Emin, r.Emax, color=RANGE_COLOR, alpha=0.22, lw=0)
            if self.show_fit.get():
                Efit = np.linspace(r.Emin, r.Emax, 200)
                ax.plot(Efit, r.C * np.exp(-Efit / r.T), color=FIT_COLOR, ls="--")
            if self.show_annot.get():
                ax.text(0.03, 0.06,
                        rf"$T = {r.T:.1f} \pm {r.T_err:.1f}\,\mathrm{{MeV}}$"
                        "\n"
                        rf"fit: ${r.Emin:.0f}$--${r.Emax:.0f}\,\mathrm{{MeV}}$,"
                        rf" $R^2 = {r.r_squared:.3f}$",
                        color=FIT_COLOR, transform=ax.transAxes)

        ax.set_xscale(self.xscale.get())
        ax.set_yscale(self.yscale.get())
        ax.set_xlabel(self.xlabel.get())
        ax.set_ylabel(self.ylabel.get())
        if self.title_txt.get().strip():
            ax.set_title(self.title_txt.get())

        self._apply_limits(ax, E[m], c[m])
        if self.minor_ticks.get():
            ax.minorticks_on()
        if self.show_legend.get() and ax.get_legend_handles_labels()[0]:
            ax.legend()

        self.fig.tight_layout()
        self.canvas.draw()

    def _apply_limits(self, ax, E, c):
        r = self.result
        xmin = self._optional(self.xmin)
        xmax = self._optional(self.xmax)
        if xmax is None and r is not None and r.ok:
            xmax = min(E.max(), 1.618 * r.Emax)  # keep the tail from being a sliver
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        ymin = self._optional(self.ymin)
        ymax = self._optional(self.ymax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

    # ------------------------------------------------------------------ output

    def save_figure(self):
        if self.energy is None:
            messagebox.showinfo("Nothing to save", "Plot a spectrum first.")
            return
        name = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=".png",
            initialfile="spectrum_fit.png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                       ("EPS", "*.eps"), ("All files", "*")],
        )
        if not name:
            return
        try:
            self.fig.savefig(name, dpi=self.save_dpi.get(), bbox_inches="tight",
                             facecolor="white")
        except Exception as exc:
            messagebox.showerror("Could not save figure", str(exc))
            return
        self._set_status(f"Saved {name} at {self.save_dpi.get()} dpi.")

    def save_report(self):
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Run a fit first.")
            return
        name = filedialog.asksaveasfilename(
            title="Save fit report", defaultextension=".txt",
            initialfile="fit_report.txt",
            filetypes=[("Text", "*.txt"), ("All files", "*")])
        if not name:
            return
        lines = [f"file      = {self.path.get()}", self.result.summary(), "",
                 "settings:",
                 f"  n_target={self.n_target.get()} "
                 f"min_decades={self.min_decades.get()} "
                 f"max_chi2={self.max_chi2.get()} "
                 f"max_curvature={self.max_curvature.get()}",
                 f"  min_bins={self.min_bins.get()} "
                 f"sigma_floor={self.sigma_floor.get()} "
                 f"min_occupancy={self.min_occupancy.get()}",
                 f"  Emin_floor={self.emin_floor.get() or 'none'} "
                 f"Emax_ceil={self.emax_ceil.get() or 'none'}"]
        try:
            with open(name, "w") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as exc:
            messagebox.showerror("Could not save report", str(exc))
            return
        self._set_status(f"Saved {name}.")

    def _set_status(self, text):
        self.status.set(text)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    app = SpectrumFitGUI(argv[0] if argv else None)
    if argv:
        app.after(100, app.fit_and_plot)
    app.mainloop()


if __name__ == "__main__":
    main()
