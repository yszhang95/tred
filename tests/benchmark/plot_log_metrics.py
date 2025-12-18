#!/usr/bin/env python3
"""Extract metrics from output_10x10.log and generate diagnostic plots."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


LOG_PATH = Path(__file__).with_name("output_10x10.log")
# LOG_PATH = Path(__file__).with_name("output_10x10_300_50.log")
LOG_PATH = Path(__file__).with_name("output_10x10_100_10.log")
LOG_PATH = Path(__file__).with_name("output_10x10_100_50.log")


def parse_tpc_blocks(lines: List[str]) -> List[Tuple[int, float, float]]:
    """Return (n_segments, elapsed_sec, peak_memory_mb) tuples."""
    peak_pattern = re.compile(r"INFO Peak cuda usage: ([0-9.]+) MB")
    itpc_pattern = re.compile(
        r"INFO itpc\d+, .*?N segments (\d+), .*?elapsed ([0-9.eE+-]+) sec"
    )
    parsed: List[Tuple[int, float, float]] = []
    pending_peak: float | None = None

    for line in lines:
        peak_match = peak_pattern.search(line)
        if peak_match:
            pending_peak = float(peak_match.group(1))
            continue

        itpc_match = itpc_pattern.search(line)
        if itpc_match and pending_peak is not None:
            parsed.append(
                (int(itpc_match.group(1)), float(itpc_match.group(2)), pending_peak)
            )
            pending_peak = None

    return parsed


def parse_summary(lines: List[str]) -> List[Tuple[str, float]]:
    """Return (label, elapsed_sec) tuples for the final summary block."""
    last_itpc = max(i for i, line in enumerate(lines) if "INFO itpc" in line)
    summary: List[Tuple[str, float]] = []

    for line in lines[last_itpc + 1 :]:
        if "Total elapsed time" in line:
            break
        if not line.startswith("INFO "):
            continue

        content = line[5:]
        if " (" not in content:
            continue
        text = content.split(" (", 1)[0].strip()
        parts = text.split(" ", 1)
        try:
            value = float(parts[0])
        except (ValueError, IndexError):
            continue
        label = parts[1].strip() if len(parts) > 1 else ""
        summary.append((label, value))

    return summary


def ensure_data():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing log file: {LOG_PATH}")
    lines = LOG_PATH.read_text().splitlines()
    tpc_data = parse_tpc_blocks(lines)
    summary_data = parse_summary(lines)
    if not tpc_data:
        raise RuntimeError("No TPC blocks found in log.")
    if not summary_data:
        raise RuntimeError("No summary block found in log.")
    return tpc_data, summary_data


def plot_trends(tpc_data: List[Tuple[int, float, float]]) -> None:
    """Generate the peak-memory and runtime vs. N segments plots."""
    nsegments = [entry[0] for entry in tpc_data]
    elapsed = [entry[1] for entry in tpc_data]
    peak_mem = [entry[2] for entry in tpc_data]
    order = sorted(range(len(nsegments)), key=lambda idx: nsegments[idx])
    nsegments_sorted = [nsegments[i] for i in order]
    elapsed_sorted = [elapsed[i] for i in order]
    peak_sorted = [peak_mem[i] for i in order]

    plt.figure(figsize=(7, 4))
    plt.plot(nsegments_sorted, peak_sorted, marker="o")
    plt.xlabel("N segments")
    plt.ylabel("Peak CUDA memory (MB)")
    plt.title("Peak memory vs. segmentation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("peak_memory_vs_nseg.png", dpi=150)

    plt.figure(figsize=(7, 4))
    plt.plot(nsegments_sorted, elapsed_sorted, marker="o", color="tab:orange")
    plt.xlabel("N segments")
    plt.ylabel("Elapsed time (s)")
    plt.title("Runtime vs. segmentation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("runtime_vs_nseg.png", dpi=150)


def plot_summary(summary_data: List[Tuple[str, float]]) -> None:
    labels = [label for label, _ in summary_data]
    values = [value for _, value in summary_data]
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("End-to-end stage breakdown")
    plt.tight_layout()
    plt.savefig("final_stage_breakdown.png", dpi=150)


def main() -> None:
    tpc_data, summary_data = ensure_data()
    plot_trends(tpc_data)
    plot_summary(summary_data)
    print(f"Parsed {len(tpc_data)} TPC blocks.")
    print(f"Summary stages: {len(summary_data)} entries.")
    print("Generated peak_memory_vs_nseg.png, runtime_vs_nseg.png, final_stage_breakdown.png")


if __name__ == "__main__":
    main()
