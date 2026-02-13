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


def get_label(npz_file: Path) -> str:
    fname = npz_file.name
    if fname.endswith(".npz"):
        fname = fname[:-4]
    if "xlimit" in fname:
        try:
            val = int(fname.split("xlimit")[-1])
            if val > 1000:
                return "No chunking"
            else:
                return f"chunk size = {val}"
        except Exception:
            pass
    return fname


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mem vs desired_mem from hardcoded .npz files.")
    parser.add_argument(
        "--save", type=Path, help="Optional path to save the figure instead of showing it"
        )
    parser.add_argument("--title", default="mem vs desired_mem", help="Plot title")
    args = parser.parse_args()

    npz_files = [
        Path("effqmem_xlimit2.npz"),
        Path("effqmem_xlimit10.npz"),
        Path("effqmem_xlimit10000.npz"),
    ]

    plt.figure(figsize=(8, 6))
    all_desired = []
    all_mem = []

    for idx, npz_file in enumerate(npz_files):
        desired_vals, mem_vals, missing = collect_pairs(npz_file)
        print(f"{npz_file}: matched {desired_vals.size} points"
              + (f", missing {len(missing)} mem arrays" if missing else ""))
        if desired_vals.size == 0:
            continue
        all_desired.append(desired_vals)
        all_mem.append(mem_vals)
        label = get_label(npz_file)
        plt.scatter(
            desired_vals,
            mem_vals,
            s=12,
            alpha=0.65,
            label=f"{label}",
        )

    if not all_desired:
        raise SystemExit("No desired/mem pairs found in the provided NPZ files.")

    stacked_desired = np.concatenate(all_desired)
    stacked_mem = np.concatenate(all_mem)
    lower = min(stacked_desired.min(), stacked_mem.min())
    upper = max(stacked_desired.max(), stacked_mem.max())
    plt.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="y = x")
    plt.xlabel("Estimated memory without additional chunking (MB)", fontsize=14)
    plt.ylabel("Measured maximal GPU memory usage (MB)", fontsize=14)
    # plt.title(args.title)
    plt.legend(fontsize=14)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
        print(f"Figure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
