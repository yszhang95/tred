#!/usr/bin/env python3
"""
Plot mem vs desired_mem from one or more .npz files.

Example:
    python plot_mem_vs_desired.py effqmem_xlimit2.npz effqmem_xlimit10.npz \
        effqmem_xlimit10000.npz --save mem_vs_desired.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIXUPS = (("itpc", "tpc"), ("ibatch", "batch"))


def normalize_suffix(name: str, prefix: str) -> str:
    suffix = name[len(prefix):].lstrip("_")
    for src, dst in FIXUPS:
        suffix = suffix.replace(src, dst)
    return suffix


def collect_pairs(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(npz_path)
    desired: Dict[str, np.ndarray] = {}
    actual: Dict[str, np.ndarray] = {}

    for key in data.files:
        if key.startswith("desired_mem"):
            desired[normalize_suffix(key, "desired_mem")] = data[key].ravel()
        elif key.startswith("mem_"):
            actual[normalize_suffix(key, "mem")] = data[key].ravel()

    miss: List[str] = []
    desired_vals: List[np.ndarray] = []
    mem_vals: List[np.ndarray] = []

    for suffix, want in desired.items():
        have = actual.get(suffix)
        if have is None:
            miss.append(suffix)
            continue
        if want.shape != have.shape:
            raise ValueError(
                f"{npz_path}: shape mismatch for {suffix}: {want.shape} vs {have.shape}"
            )
        desired_vals.append(want)
        mem_vals.append(have)

    data.close()
    if not desired_vals:
        return np.empty(0), np.empty(0), miss
    return np.concatenate(desired_vals), np.concatenate(mem_vals), miss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_files", nargs="+", type=Path, help="NPZ files to inspect")
    parser.add_argument(
        "--save", type=Path, help="Optional path to save the figure instead of showing it"
        )
    parser.add_argument("--title", default="mem vs desired_mem", help="Plot title")
    args = parser.parse_args()

    plt.figure(figsize=(7, 7))
    all_desired = []
    all_mem = []

    for idx, npz_file in enumerate(args.npz_files):
        desired_vals, mem_vals, missing = collect_pairs(npz_file)
        print(f"{npz_file}: matched {desired_vals.size} points"
              + (f", missing {len(missing)} mem arrays" if missing else ""))
        if desired_vals.size == 0:
            continue
        all_desired.append(desired_vals)
        all_mem.append(mem_vals)
        plt.scatter(
            desired_vals,
            mem_vals,
            s=12,
            alpha=0.65,
            label=f"{npz_file.name} (n-test-points={desired_vals.size})",
        )

    if not all_desired:
        raise SystemExit("No desired/mem pairs found in the provided NPZ files.")

    stacked_desired = np.concatenate(all_desired)
    stacked_mem = np.concatenate(all_mem)
    lower = min(stacked_desired.min(), stacked_mem.min())
    upper = max(stacked_desired.max(), stacked_mem.max())
    plt.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="y = x")
    plt.xlabel("desired_mem")
    plt.ylabel("mem")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
        print(f"Figure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

