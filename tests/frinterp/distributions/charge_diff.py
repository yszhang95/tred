#!/usr/bin/env python
"""
Quantify the charge difference between the 0.05 us and 0.10 us outputs,
expressed as a percentage of hits.

Two views:
  (1) per-hit: match hits one-to-one by (tpc, pixel x, pixel y, arrival time)
      and compare recorded charge for the matched pairs; report unmatched %.
  (2) distribution-level: total-variation distance of the charge spectra
      (= % of hits that would have to change bin to make them identical).
"""
import os
import re
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
TOL_US = 0.6  # arrival-time match tolerance (< 1.2 us ADC down-time => no ambiguity)


def load(fn, tspace):
    d = np.load(os.path.join(DATA, fn), allow_pickle=True)
    rows = defaultdict(list)  # (tpc, pix_x, pix_y) -> list of [time_us, charge]
    for k in d.files:
        m = re.match(r"hits_tpc(\d+)_batch(\d+)_location$", k)
        if not m:
            continue
        t = int(m.group(1))
        loc = d[k]
        if loc.shape[0] == 0:
            continue
        hit = d[k[: -len("_location")]]
        for i in range(loc.shape[0]):
            rows[(t, int(loc[i, 0]), int(loc[i, 1]))].append(
                [loc[i, 2] * tspace, float(hit[i, 3])]
            )
    return rows


def match(a, b, tol=TOL_US):
    """Greedy nearest-time match within each pixel. Returns (qa, qb) arrays of
    matched charges, plus unmatched counts."""
    qa_m, qb_m = [], []
    only_a = only_b = 0
    keys = set(a) | set(b)
    for key in keys:
        la = sorted(a.get(key, []))
        lb = sorted(b.get(key, []))
        used_b = set()
        for ta, qa in la:
            best, bestdt = -1, tol
            for j, (tb, qb) in enumerate(lb):
                if j in used_b:
                    continue
                dt = abs(ta - tb)
                if dt < bestdt:
                    best, bestdt = j, dt
            if best >= 0:
                used_b.add(best)
                qa_m.append(qa)
                qb_m.append(lb[best][1])
            else:
                only_a += 1
        only_b += len(lb) - len(used_b)
    return np.array(qa_m), np.array(qb_m), only_a, only_b


def main():
    a = load("output_10x10_0p05us.npz", 0.05)
    b = load("output_10x10_0p1us.npz", 0.10)
    na = sum(len(v) for v in a.values())
    nb = sum(len(v) for v in b.values())

    qa, qb, only_a, only_b = match(a, b)
    nmatch = len(qa)
    print(f"hits: 0.05us N={na}, 0.10us N={nb}")
    print(f"matched pairs            : {nmatch}")
    print(f"only in 0.05us (unmatched): {only_a}  = {100*only_a/na:.2f}% of 0.05us hits")
    print(f"only in 0.10us (unmatched): {only_b}  = {100*only_b/nb:.2f}% of 0.10us hits")
    print(f"matched fraction         : {100*nmatch/na:.2f}% of 0.05us, "
          f"{100*nmatch/nb:.2f}% of 0.10us")

    # per-hit relative charge difference (reference = 0.05us value)
    rel = np.abs(qb - qa) / qa
    print("\nper-hit |dQ|/Q on matched hits (ref = 0.05us):")
    print(f"  median = {100*np.median(rel):.3f}%   mean = {100*rel.mean():.3f}%   "
          f"95th pct = {100*np.percentile(rel,95):.2f}%   max = {100*rel.max():.2f}%")
    for thr in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        frac = 100 * np.mean(rel <= thr)
        print(f"  within {thr*100:5.1f}% : {frac:6.2f}% of matched hits")

    # signed bias
    sig = (qb - qa) / qa
    print(f"  signed mean (0.10-0.05)/0.05 = {100*sig.mean():+.3f}%  "
          f"(median {100*np.median(sig):+.3f}%)")

    # distribution-level total variation on charge spectrum
    lo = min(qa.min() if nmatch else 0, 0)
    allq = np.concatenate([np.concatenate(list(a.values()))[:, 1],
                           np.concatenate(list(b.values()))[:, 1]])
    bins = np.linspace(allq.min(), allq.max(), 200)
    qa_all = np.concatenate(list(a.values()))[:, 1]
    qb_all = np.concatenate(list(b.values()))[:, 1]
    ha, _ = np.histogram(qa_all, bins=bins, density=False)
    hb, _ = np.histogram(qb_all, bins=bins, density=False)
    pa = ha / ha.sum()
    pb = hb / hb.sum()
    tv = 0.5 * np.abs(pa - pb).sum()
    print(f"\ndistribution-level charge spectrum:")
    print(f"  total-variation distance = {100*tv:.2f}% of hits "
          f"(fraction differing between the two normalized spectra)")
    print(f"  summed charge: 0.05us={qa_all.sum():.0f} ke, 0.10us={qb_all.sum():.0f} ke, "
          f"diff={100*(qb_all.sum()-qa_all.sum())/qa_all.sum():+.3f}%")


if __name__ == "__main__":
    main()
