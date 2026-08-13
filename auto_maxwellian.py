"""
Automatic detection of the exponential ("straight on a log plot") portion of a
particle energy spectrum, plus a temperature fit over that portion.

Drop-in extension of Maxwellian_fit (github.com/StevE-Ong/Maxwellian_fit): same
physics, dN/dE = C exp(-E/T), but Emin and Emax are found from the data rather
than supplied by hand.

Why the hand-set range fails
----------------------------
A PIC electron spectrum has four regions on a log plot:

    (1) a steep cold bulk at low E,
    (2) a knee,
    (3) the hot exponential tail   <-- the only genuinely straight part,
    (4) a sparse high-E floor where bins hold one or two macroparticles.

Starting the fit at a low threshold drags the bulk into the regression: the
intercept is pulled up and the slope flattened, and the fitted line ends up
sitting above the entire spectrum. Ending too high fits the sparse floor.

Method
------
1. Merge adjacent bins to suppress single-macroparticle spikes without
   smoothing away real curvature.
2. Estimate the log-space scatter *empirically*, as a running MAD about a
   running median. This is the important bit: a weighted PIC histogram has no
   fixed count quantum -- macroparticle weights vary continuously with the
   initial local density -- so the Poisson error of a bin cannot be recovered
   from its value. Measuring the scatter instead makes the detector
   self-calibrating, and works equally on experimental spectra.
3. Scan every candidate window, keep those whose log-residuals are flat to
   within that measured noise, and take the one spanning the widest energy
   range. The widest straight stretch is the tail; a min_decades guard stops a
   flat noise floor from qualifying, and the bulk loses because it is steep but
   narrow.

`local_temperature` exposes the sliding-window T_loc(E); its plateau is the
tail, and plotting it is the fastest way to see whether a range is sane.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FitResult", "auto_maxwellian_fit", "local_temperature", "Maxwellian_fit_auto"]

LN10 = np.log(10.0)


class FitResult:
    """Outcome of a temperature fit, with the diagnostics needed to trust it."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        if not self.ok:
            return f"<FitResult FAILED: {self.message}>"
        return (
            f"<FitResult T={self.T:.3f}+/-{self.T_err:.3f} MeV "
            f"range=[{self.Emin:.2f},{self.Emax:.2f}] MeV "
            f"decades={self.decades:.2f} R2={self.r_squared:.4f}"
            + (f" WARN:{self.message}" if self.message else "")
            + ">"
        )

    def summary(self) -> str:
        if not self.ok:
            return f"fit failed: {self.message}"
        lines = [
            f"T     = {self.T:.3f} +/- {self.T_err:.3f} MeV",
            f"fit range = [{self.Emin:.3f}, {self.Emax:.3f}] MeV   (auto-detected)",
            f"falloff   = {self.decades:.2f} decades over {self.n_bins} bins",
            f"R^2       = {self.r_squared:.4f}    chi2/dof = {self.chi2_red:.2f}",
            f"amplitude = {self.C:.4g}",
            f"log noise = {self.sigma_log:.3f} (measured)",
        ]
        if self.message:
            lines.append(f"WARNING: {self.message}")
        return "\n".join(lines)


def _centers_and_widths(energy):
    E = np.asarray(energy, dtype=float).ravel()
    if E.size < 2:
        raise ValueError("need at least 2 energy bins")
    d = np.diff(E)
    if np.any(d <= 0):
        raise ValueError("energy must be strictly increasing")
    w = np.empty_like(E)
    w[1:-1] = 0.5 * (d[:-1] + d[1:])
    w[0], w[-1] = d[0], d[-1]
    return E, w


def _merge(E, width, N, k):
    """Merge groups of k adjacent bins; drop groups that end up empty."""
    n = (E.size // k) * k
    Eg = E[:n].reshape(-1, k)
    Wg = width[:n].reshape(-1, k)
    Ng = N[:n].reshape(-1, k)
    tot = Ng.sum(axis=1)
    wid = Wg.sum(axis=1)
    keep = tot > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        cen = (Eg * Ng).sum(axis=1) / np.where(tot > 0, tot, 1.0)
    return cen[keep], (tot / wid)[keep], tot[keep]


def _running(y, half, func):
    out = np.empty_like(y)
    n = y.size
    for i in range(n):
        s = slice(max(0, i - half), min(n, i + half + 1))
        out[i] = func(y[s])
    return out


def _empirical_sigma(y, half=8, floor=0.05):
    """Local log-space scatter: MAD about a running median, itself smoothed."""
    base = _running(y, half, np.median)
    resid = y - base
    mad = _running(np.abs(resid), half, np.median) * 1.4826
    mad = _running(mad, half, np.mean)          # smooth the noise estimate
    return np.maximum(mad, floor)


def _curvature_significance(x, y, w):
    """|c|/sigma_c for a weighted quadratic fit y = a + b*x + c*x^2.

    The cold bulk and the sparse high-E floor both enter a window as systematic
    *curvature*, which a chi-squared ceiling tolerates by simply inflating the
    scatter. Testing the quadratic term directly is what actually keeps the knee
    out of the fit.
    """
    xc = x - x.mean()
    A = np.vstack([np.ones_like(xc), xc, xc ** 2]).T
    Aw = A * w[:, None]
    M = A.T @ Aw
    try:
        Minv = np.linalg.inv(M)
        coef = Minv @ (Aw.T @ y)
    except np.linalg.LinAlgError:
        return np.inf
    var_c = Minv[2, 2]
    if not np.isfinite(var_c) or var_c <= 0:
        return np.inf
    return abs(coef[2]) / np.sqrt(var_c)


def _wls(x, y, w):
    """Weighted least squares. Returns slope, intercept, chi2, weighted TSS."""
    sw = w.sum()
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    sxx = (w * (x - mx) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan, np.inf, 0.0
    b = (w * (x - mx) * (y - my)).sum() / sxx
    a = my - b * mx
    r = y - (a + b * x)
    return b, a, float((w * r ** 2).sum()), float((w * (y - my) ** 2).sum())


def _occupancy_cutoff(N, min_occupancy=0.5, win=25):
    """Highest index at which the histogram is still a continuous distribution.

    Above some energy a PIC spectrum stops being a distribution and becomes a
    scatter of individual macroparticles: most bins are empty and the occupied
    ones hold one particle each. Merging such a region produces densities set by
    the merge width rather than by physics, which then reads as a long straight
    decline and can dominate a fit. Cutting where local occupancy falls below
    `min_occupancy` removes it without needing to know particle weights.
    """
    occ = (N > 0).astype(float)
    n = occ.size
    if n < win:
        return n
    kern = np.ones(win) / win
    frac = np.convolve(occ, kern, mode="same")
    good = np.where(frac >= min_occupancy)[0]
    if good.size == 0:
        return n
    return int(good[-1]) + 1


def _prepare(energy, counts, n_target, Emin_floor, Emax_ceil, sigma_floor,
             min_occupancy=0.5):
    E, width = _centers_and_widths(energy)
    c = np.asarray(counts, dtype=float).ravel()
    n = min(E.size, c.size)
    E, width, c = E[:n], width[:n], c[:n]

    good = np.isfinite(c) & (c > 0) & np.isfinite(E)
    if Emin_floor is not None:
        good &= E >= Emin_floor
    if Emax_ceil is not None:
        good &= E <= Emax_ceil
    if good.sum() < 10:
        return None

    N = np.where(good, c, 0.0) * width
    cut = _occupancy_cutoff(N, min_occupancy)
    E, width, N = E[:cut], width[:cut], N[:cut]
    if np.count_nonzero(N) < 10:
        return None
    occupied = int(np.count_nonzero(N))
    k = max(1, int(round(occupied / max(n_target, 5))))
    Eb, dens, tot = _merge(E, width, N, k)
    if Eb.size < 8:
        Eb, dens, tot = _merge(E, width, N, 1)
    y = np.log(dens)
    sigma = _empirical_sigma(y, floor=sigma_floor)
    return Eb, dens, y, sigma, k


def local_temperature(energy, counts, n_target=150, window=7,
                      Emin_floor=None, Emax_ceil=None, sigma_floor=0.05,
                      min_occupancy=0.5):
    """Sliding-window local temperature T_loc(E). Its plateau is the hot tail.

    Returns (E, T_loc, T_loc_err); entries are NaN where the local slope is not
    negative or the window runs off the end.
    """
    prep = _prepare(energy, counts, n_target, Emin_floor, Emax_ceil, sigma_floor,
                    min_occupancy)
    if prep is None:
        return np.array([]), np.array([]), np.array([])
    Eb, dens, y, sigma, _ = prep
    h = max(1, window // 2)
    T = np.full(Eb.size, np.nan)
    Terr = np.full(Eb.size, np.nan)
    w = 1.0 / sigma ** 2
    for i in range(h, Eb.size - h):
        s = slice(i - h, i + h + 1)
        b, _, chi2, _ = _wls(Eb[s], y[s], w[s])
        if np.isfinite(b) and b < 0:
            T[i] = -1.0 / b
            xs, ws = Eb[s], w[s]
            mx = (ws * xs).sum() / ws.sum()
            sxx = (ws * (xs - mx) ** 2).sum()
            if sxx > 0:
                T[i] = -1.0 / b
                Terr[i] = T[i] ** 2 * np.sqrt(1.0 / sxx)
    return Eb, T, Terr


def auto_maxwellian_fit(
    energy,
    counts,
    n_target=150,
    min_decades=1.0,
    max_chi2=3.0,
    max_curvature=3.0,
    min_bins=8,
    min_hot_fraction=None,
    Emin_floor=None,
    Emax_ceil=None,
    sigma_floor=0.05,
    min_occupancy=0.5,
):
    """Fit dN/dE = C exp(-E/T), detecting the exponential range automatically.

    Parameters
    ----------
    energy, counts : 1D arrays
        Bin energy in MeV (centers or left edges, spacing may be non-uniform)
        and dN/dE (or raw counts if the bins are uniform).
    n_target : int
        Approximate number of bins after merging. Lower it for noisy spectra.
    min_decades : float
        Least falloff an accepted window must span. Guards against locking onto
        a flat, sparse high-energy floor.
    max_chi2 : float
        Straightness tolerance, as reduced chi-squared against the *measured*
        log-space noise. Raise to accept a looser exponential.
    max_curvature : float
        Reject a window if the quadratic term of a parabola fit is significant
        at more than this many sigma. This is what keeps the cold bulk and the
        sparse high-E floor out; a chi-squared ceiling alone will happily
        swallow the knee and absorb it as extra scatter.
    min_bins : int
        Least merged bins in an accepted window.
    min_hot_fraction : float or None
        Reject the fit if the detected window holds less than this *number*
        fraction of the spectrum. Off by default, and deliberately so: the hot
        tail is always a small fraction of the electron count (the cold bulk
        dominates), so typical healthy values are a few percent. Whether a
        snapshot has a hot population at all is not something the fitter can
        decide without an external energy scale -- use your own f_hot threshold
        for that. `window_fraction` is reported either way.
    Emin_floor, Emax_ceil : float, optional
        Hard limits on the search, if some region must be excluded a priori.
    sigma_floor : float
        Lower bound on the estimated log-space noise, so that a locally smooth
        stretch cannot drive the tolerance to zero.
    min_occupancy : float
        Discard everything above the energy where the fraction of populated
        original bins drops below this. That is where the spectrum stops being a
        distribution and becomes individual macroparticles.

    Returns
    -------
    FitResult
    """
    fail = dict(ok=False, T=np.nan, T_err=np.nan, Emin=np.nan, Emax=np.nan,
                decades=0.0, r_squared=np.nan, chi2_red=np.nan, C=np.nan,
                n_bins=0, sigma_log=np.nan, E_rebin=np.array([]),
                dNdE_rebin=np.array([]), slope=np.nan, intercept=np.nan,
                N_window=0.0, N_total=0.0, window_fraction=0.0)

    prep = _prepare(energy, counts, n_target, Emin_floor, Emax_ceil, sigma_floor,
                    min_occupancy)
    if prep is None:
        return FitResult(message="fewer than 10 populated bins", **fail)
    Eb, dens, y, sigma, k = prep
    if Eb.size < min_bins:
        return FitResult(message=f"only {Eb.size} bins after merging", **fail)

    w = 1.0 / sigma ** 2
    nb = Eb.size
    best = None
    for s in range(nb - min_bins + 1):
        for e in range(s + min_bins, nb + 1):
            b, a, chi2, _ = _wls(Eb[s:e], y[s:e], w[s:e])
            if not np.isfinite(b) or b >= 0:
                continue
            m = e - s
            if chi2 / (m - 2) > max_chi2:
                continue
            span = Eb[e - 1] - Eb[s]
            decades = -b * span / LN10
            if decades < min_decades:
                continue
            # The fitted falloff must be backed by real dynamic range in the
            # data. Merging drops empty bins, so a window can span a wide energy
            # range while holding only a few populated bins at its far end; the
            # extrapolated "decades" is then fictitious (values far exceeding
            # the spectrum's actual dynamic range are the tell).
            observed = (y[s:e].max() - y[s:e].min()) / LN10
            if decades > observed + 1.0:
                continue
            if _curvature_significance(Eb[s:e], y[s:e], w[s:e]) > max_curvature:
                continue
            if best is None or span > best[0]:
                best = (span, s, e, b, a, chi2)

    if best is None:
        return FitResult(
            message="no window is straight to within tolerance and spans "
                    f">{min_decades} decades; relax max_chi2/min_decades or "
                    "inspect the spectrum", **fail)

    span, s, e, b, a, chi2 = best
    xs, ys, ws = Eb[s:e], y[s:e], w[s:e]
    m = xs.size
    T = -1.0 / b
    mx = (ws * xs).sum() / ws.sum()
    sxx = (ws * (xs - mx) ** 2).sum()
    T_err = T ** 2 * np.sqrt(1.0 / sxx)
    _, _, _, tss = _wls(xs, ys, ws)
    r2 = 1.0 - chi2 / tss if tss > 0 else np.nan
    decades = -b * span / LN10

    # how much of the spectrum the fit actually describes
    N_all = float(np.sum(dens * np.gradient(Eb))) if Eb.size > 1 else 0.0
    sel = (Eb >= xs[0]) & (Eb <= xs[-1])
    N_win = float(np.sum((dens * np.gradient(Eb))[sel])) if Eb.size > 1 else 0.0
    frac = N_win / N_all if N_all > 0 else 0.0

    msg = ""
    if min_hot_fraction is not None and frac < min_hot_fraction:
        return FitResult(
            message=f"fit window holds only {frac:.1%} of the spectrum "
                    f"(< min_hot_fraction={min_hot_fraction:.1%}); there is no "
                    "populated hot tail in this snapshot", **fail)
    if s == 0:
        msg = "window starts at the lowest populated bin; the cold bulk may not be excluded"
    elif decades < 1.5:
        msg = f"only {decades:.2f} decades of falloff; T is weakly constrained"

    return FitResult(
        ok=True, T=T, T_err=T_err, Emin=float(xs[0]), Emax=float(xs[-1]),
        decades=decades, r_squared=r2, chi2_red=chi2 / (m - 2), C=float(np.exp(a)),
        n_bins=m, sigma_log=float(np.mean(sigma[s:e])), message=msg,
        decades_observed=float((ys.max() - ys.min()) / LN10),
        E_rebin=Eb, dNdE_rebin=dens, slope=b, intercept=a, merge_factor=k,
        N_window=N_win, N_total=N_all, window_fraction=frac,
    )


def Maxwellian_fit_auto(counts, energy, **kw):
    """Signature-compatible with Maxwellian_fit, minus Emin/Emax.

    Returns temperature in MeV (NaN on failure) so it can be swapped in directly.
    """
    return auto_maxwellian_fit(energy, counts, **kw).T


def plot_fit(energy, counts, result=None, ax=None, show_local_T=True,
             label=None, **fit_kw):
    """Spectrum, detected window and fit on one axes, for eyeballing the range.

    The shaded band is the auto-detected fit range. The twin axis shows the
    local temperature T_loc(E); a flat stretch there is the exponential tail,
    and the shaded band should sit on it.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.2))
    if result is None:
        result = auto_maxwellian_fit(energy, counts, **fit_kw)

    E = np.asarray(energy, float).ravel()
    c = np.asarray(counts, float).ravel()
    n = min(E.size, c.size)
    E, c = E[:n], c[:n]
    m = c > 0
    ax.semilogy(E[m], c[m], lw=0.6, color="0.75", label="raw", zorder=1)
    if result.ok and result.E_rebin.size:
        ax.semilogy(result.E_rebin, result.dNdE_rebin, lw=1.3, color="#1f3b73",
                    label=f"merged (x{getattr(result,'merge_factor','?')})", zorder=2)

    if result.ok:
        ax.axvspan(result.Emin, result.Emax, color="#f5c518", alpha=0.22,
                   zorder=0, label="auto range")
        xf = np.linspace(result.Emin, result.Emax, 100)
        ax.semilogy(xf, np.exp(result.intercept + result.slope * xf), "r--", lw=1.8,
                    zorder=4, label=f"$T$ = {result.T:.2f} $\\pm$ {result.T_err:.2f} MeV")
        ax.set_xlim(0, min(E[m].max(), result.Emax * 1.6))
        lo = result.dNdE_rebin.min()
        ax.set_ylim(max(lo * 0.3, result.dNdE_rebin.max() * 1e-7),
                    result.dNdE_rebin.max() * 3)
    else:
        ax.set_title("fit failed", color="firebrick", fontsize=9)

    if show_local_T:
        Eloc, Tloc, _ = local_temperature(energy, counts)
        if Eloc.size:
            ax2 = ax.twinx()
            ax2.plot(Eloc, Tloc, color="#2e8b57", lw=1.0, alpha=0.85)
            ax2.set_ylabel(r"$T_{\rm loc}$ (MeV)", color="#2e8b57", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="#2e8b57", labelsize=8)
            good = np.isfinite(Tloc)
            if good.any():
                ax2.set_ylim(0, np.nanpercentile(Tloc[good], 95) * 1.6)

    ax.set_xlabel(r"$E$ (MeV)")
    ax.set_ylabel(r"$dN/dE$ (arb.)")
    if label:
        ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    return ax, result
