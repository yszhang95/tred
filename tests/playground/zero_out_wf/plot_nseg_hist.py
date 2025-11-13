#!/usr/bin/env python3
"""
Histogram the Nseg_itpcX_ibatchY values from NPZ files.

Example:
    python plot_nseg_hist.py full.npz effqmem_xlimit10.npz --bins 50 \
        --save nseg_hist.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Accept both itpc/tpc and ibatch/batch for flexibility.
_key_fixups = (
    ("itpc", "tpc"),
    ("ibatch", "batch"),
)
# Match keys without chunk suffixes only.
_KEY_RE = re.compile(r"^Nseg_(?:i?tpc\\d+)_i?batch\\d+$")


def _normalize(key: str) -> str:
    for src, dst in _key_fixups:
        key = key.replace(src, dst)
    return key


def collect_nseg_values(path: Path) -> np.ndarray:
    with np.load(path) as data:
        values: List[np.ndarray] = []
        missing_chunks: Dict[str, str] = {}
        for key in data.files:
            if not key.startswith("Nseg_"):
                continue
            if "_ichunk" in key:
                # keep note only the first time we see a chunked variant
                base = key.split("_ichunk", 1)[0]
                missing_chunks.setdefault(base, key)
                continue
            normalized = _normalize(key)
            if not _KEY_RE.match(normalized):
                continue
            arr = np.asarray(data[key]).ravel()
            if arr.size:
                values.append(arr)

        if not values:
            chunk_msg = (
                " (only chunked keys found)" if missing_chunks else ""
            )
            raise ValueError(f"{path} contains no Nseg_itpcX_ibatchY arrays{chunk_msg}")

    stacked = np.concatenate(values)
    print(f"{path}: loaded {stacked.size} Nseg entries from {len(values)} keys")
    if missing_chunks:
        print(f"  Skipped {len(missing_chunks)} chunked keys (use --allow-chunk to include later).")
    return stacked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_files", nargs="+", type=Path)
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins")
    parser.add_argument("--save", type=Path, help="Path to save the figure instead of showing it")
    parser.add_argument("--title", default="Histogram of Nseg_itpcX_ibatchY")
    args = parser.parse_args()

    all_values = []
    for npz_file in args.npz_files:
        all_values.append(collect_nseg_values(npz_file))

    combined = np.concatenate(all_values)
    plt.figure(figsize=(8, 5))
    plt.hist(combined, bins=args.bins, color="#1f77b4", alpha=0.75, edgecolor="black")
    plt.xlabel("Nseg")
    plt.ylabel("Count")
    plt.title(args.title)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
        print(f"Histogram saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
