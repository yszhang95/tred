#!/usr/bin/env python
"""
Overlay distributions of pixel x/y index, arrival time, and recorded charge
for the two graph_effq fullsim outputs (0.05 us vs 0.1 us time spacing).

Each output npz stores, per (tpc, batch):
    hits_tpc{t}_batch{b}            (N, 4): [z_drift, x_phys, y_phys, charge_ke]
    hits_tpc{t}_batch{b}_location   (N, 5): [pix_x, pix_y, cross_tick, hold_tick, start_tick]

Hits with charge > QMAX (=100 ke) are filtered out. All panels use uniform
binning, a linear y-axis, and per-bin Poisson (sqrt(N)) error bars.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)  # tests/frinterp

FILES = [
    ("output_10x10_0p05us.npz", 0.05, "C0"),
    ("output_10x10_0p1us.npz",  0.10, "C1"),
]

QMAX = 100.0  # ke-, drop hits above this
LOC_RE = re.compile(r"hits_tpc(\d+)_batch(\d+)_location$")


def collect(npz_path, tspace, qmax=QMAX):
    """Return dict of flat arrays aggregated over all tpc/batch (Q<=qmax)."""
    d = np.load(npz_path, allow_pickle=True)
    pix_x, pix_y, tick, charge = [], [], [], []
    for k in d.files:
        m = LOC_RE.match(k)
        if not m:
            continue
        loc = d[k]
        if loc.shape[0] == 0:
            continue
        hit = d[k[: -len("_location")]]
        q = hit[:, 3]
        keep = q <= qmax
        if not keep.any():
            continue
        pix_x.append(loc[keep, 0])
        pix_y.append(loc[keep, 1])
        tick.append(loc[keep, 2])
        charge.append(q[keep])
    pix_x = np.concatenate(pix_x)
    pix_y = np.concatenate(pix_y)
    tick = np.concatenate(tick).astype(np.float64)
    charge = np.concatenate(charge)
    return dict(
        pix_x=pix_x,
        pix_y=pix_y,
        tick=tick,
        time_us=tick * tspace,
        charge=charge,
        n=pix_x.size,
        tspace=tspace,
    )


def shared_bins(arrays, nbins, lo=None, hi=None):
    allv = np.concatenate(arrays)
    lo = allv.min() if lo is None else lo
    hi = allv.max() if hi is None else hi
    return np.linspace(lo, hi, nbins + 1)


def hist_poisson(ax, values, bins, color, label, dx=0.0):
    """Uniform histogram drawn as points with per-bin Poisson (sqrt N) bars.
    dx shifts the markers slightly so the two series don't overlap exactly."""
    counts, _ = np.histogram(values, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:]) + dx
    ax.errorbar(centers, counts, yerr=np.sqrt(counts), fmt="o", ms=3.0,
                lw=1.0, capsize=1.5, color=color, label=label)


def main():
    data = []
    for fname, tspace, color in FILES:
        s = collect(os.path.join(DATA, fname), tspace)
        s["color"] = color
        s["label"] = f"{tspace:g} us  (N={s['n']})"
        data.append(s)
        print(f"{fname}: N={s['n']} (Q<= {QMAX:g} ke), "
              f"pix_x[{s['pix_x'].min()},{s['pix_x'].max()}], "
              f"pix_y[{s['pix_y'].min()},{s['pix_y'].max()}], "
              f"charge[{s['charge'].min():.1f},{s['charge'].max():.1f}] ke")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "graph_effq output distributions: 0.05 us vs 0.10 us response\n"
        f"(event 1003, all TPCs/batches, Q<= {QMAX:g} ke, Poisson errors)",
        fontsize=13,
    )

    #   key,      xlabel,                            nbins, lo,   hi
    panels = [
        ("pix_x",   "pixel x index",                  40, None, None),
        ("pix_y",   "pixel y index",                  40, None, None),
        ("time_us", "arrival time [us] = tick x tspace", 60, None, None),
        ("charge",  "recorded charge [ke-]",          50, 0.0,  QMAX),
    ]

    for (key, xlabel, nbins, lo, hi), ax in zip(panels, axes.ravel()):
        bins = shared_bins([s[key] for s in data], nbins, lo=lo, hi=hi)
        bw = bins[1] - bins[0]
        for i, s in enumerate(data):
            dx = (i - (len(data) - 1) / 2) * 0.18 * bw
            hist_poisson(ax, s[key], bins, s["color"], s["label"], dx=dx)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("hits / bin  (dN)")
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "overlay_distributions.png")
    fig.savefig(out, dpi=130)
    print("saved", out)

    print("\nsummary (median / mean):")
    for s in data:
        print(f"  tspace={s['tspace']:g}: "
              f"charge med={np.median(s['charge']):.2f} mean={s['charge'].mean():.2f} ke | "
              f"sum={s['charge'].sum():.0f} ke | "
              f"time_us med={np.median(s['time_us']):.1f}")


if __name__ == "__main__":
    main()
