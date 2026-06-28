#!/usr/bin/env python
'''
Analyze point-charge fullsim output: 2D histogram of (true q, hit q) per pixel.

Matched on (iter, pix_y, pix_z):
    x = effq    : the true (effective) charge on that pixel
    y = hit q   : the readout hit charge on that pixel
A pixel that links to several hits fills the histogram once per hit.
A pixel with no hit fills once with y = 0.

Arrays:
  effq : [iter, pix_y, pix_z, effq_ke]                          (per-pixel true charge)
  hits : [iter, pix_y, pix_z, time_tick, x, y, z, charge_ke]    (per hit)

Usage:
    uv run python analyze.py pointq_8ke_10000.npz [--threshold 5] [-o out.png]
'''
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_entries(f):
    effq = f['effq']   # [iter, pix_y, pix_z, effq_ke]
    hits = f['hits']   # [iter, pix_y, pix_z, time_tick, x, y, z, charge_ke]

    # (iter, pix_y, pix_z) -> list of hit charges on that pixel
    hitmap = defaultdict(list)
    for r in hits:
        hitmap[(int(r[0]), int(r[1]), int(r[2]))].append(float(r[7]))

    xs, ys = [], []
    n_hit_fill = 0
    for r in effq:
        e = float(r[3])
        qs = hitmap.get((int(r[0]), int(r[1]), int(r[2])), [])
        if qs:
            for q in qs:                 # fill once per hit on this pixel
                xs.append(e); ys.append(q)
            n_hit_fill += len(qs)
        else:
            xs.append(e); ys.append(0.0)  # no hit on this pixel
    return np.array(xs), np.array(ys), effq.shape[0], n_hit_fill


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz')
    ap.add_argument('--threshold', type=float, default=5.0, help='hit threshold [ke-] (drawn)')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--bins', type=int, default=120)
    ap.add_argument('--qmax', type=float, default=None)
    args = ap.parse_args()

    f = np.load(args.npz)
    xs, ys, npix, n_hit_fill = build_entries(f)

    out = args.out or (args.npz.rsplit('.npz', 1)[0] + '_hist2d_effq.png')
    qmax = args.qmax if args.qmax is not None else max(xs.max(), ys.max()) * 1.1
    rng = [[0.0, qmax], [0.0, qmax]]

    fig, ax = plt.subplots(figsize=(6, 5))
    _, _, _, im = ax.hist2d(xs, ys, bins=args.bins, range=rng, cmin=1)
    ax.plot([0, qmax], [0, qmax], 'r--', lw=1, label='hit q = true q')
    ax.axhline(args.threshold, color='orange', ls=':', lw=1, label=f'threshold {args.threshold:g} ke-')
    ax.set_xlabel('total charge per pixel (effq) [ke-]')
    ax.set_ylabel('hit q [ke-]   (0 = no hit)')
    ax.set_title(f'{npix} pixels, {n_hit_fill} pixel-hits')
    ax.legend(loc='upper left', fontsize=8)
    fig.colorbar(im, ax=ax, label='entries')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'effq pixels: {npix}, pixel-hit fills: {n_hit_fill}, total entries: {len(xs)}')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
