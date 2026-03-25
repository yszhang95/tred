#!/usr/bin/env python3
"""Extract metrics from one or more logs and generate diagnostic plots."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import matplotlib.pyplot as plt


DEFAULT_LOG = Path(__file__).with_name("output_10x10.log")


class LogData(TypedDict):
    tpc: List[Tuple[int, float, float]]
    summary: List[Tuple[str, float]]


def format_label(path: Path) -> str:
    """Derive a friendly legend label from the log filename."""
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 4 and parts[-1].isdigit() and parts[-2].isdigit():
        return f"raster batch {parts[-2]}, convo batch {parts[-1]}"
    return stem


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


def parse_log_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    lines = path.read_text().splitlines()
    tpc_data = parse_tpc_blocks(lines)
    summary_data = parse_summary(lines)
    if not tpc_data:
        raise RuntimeError(f"No TPC blocks found in {path}.")
    if not summary_data:
        raise RuntimeError(f"No summary block found in {path}.")
    return tpc_data, summary_data


def plot_trend(log_data: Dict[str, LogData], metric: str) -> None:
    """Plot runtime or peak memory vs. N segments for each log."""
    ylabel = "Elapsed time (s)" if metric == "elapsed" else "Peak CUDA memory (MB)"
    title = (
        "Runtime vs. segmentation"
        if metric == "elapsed"
        else "Peak memory vs. segmentation"
    )
    filename = (
        "runtime_vs_nseg.png" if metric == "elapsed" else "peak_memory_vs_nseg.png"
    )

    plt.figure(figsize=(7, 4))
    for label, payload in log_data.items():
        tpc_data = payload["tpc"]
        nsegments = [entry[0] for entry in tpc_data]
        order = sorted(range(len(nsegments)), key=lambda idx: nsegments[idx])
        nsegments_sorted = [nsegments[i] for i in order]
        if metric == "elapsed":
            values = [tpc_data[i][1] for i in order]
        else:
            values = [tpc_data[i][2] for i in order]
        if metric == "elapsed":
            plt.plot(nsegments_sorted, values, marker="o", linestyle="-", label=label)
        else:
            plt.scatter(
                nsegments_sorted,
                values,
                label=label,
                alpha=0.75,
                edgecolors="none",
            )

    plt.xlabel("N segments")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def plot_stage_pies(log_data: Dict[str, LogData]) -> None:
    """Create side-by-side pie charts for each log's terminal summary."""
    labels = list(log_data.keys())
    count = len(labels)
    fig, axes = plt.subplots(1, count, figsize=(4.5 * count, 4.5))
    if count == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        summary = log_data[label]["summary"]
        stage_labels = [entry[0] for entry in summary]
        values = [entry[1] for entry in summary]
        ax.pie(values, labels=stage_labels, autopct="%1.1f%%", startangle=140)
        ax.set_title(label)

    fig.suptitle("End-to-end stage breakdown")
    plt.tight_layout()
    plt.savefig("stage_breakdown.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from benchmark logs.")
    parser.add_argument(
        "logs",
        nargs="*",
        default=[str(DEFAULT_LOG)],
        help="Log files to parse (default: output_10x10.log)",
    )
    args = parser.parse_args()

    log_entries: Dict[str, LogData] = {}
    for log_path_str in args.logs:
        log_path = Path(log_path_str)
        tpc_data, summary_data = parse_log_file(log_path)
        label = format_label(log_path)
        log_entries[label] = {"tpc": tpc_data, "summary": summary_data}
        print(f"{label}: {len(tpc_data)} TPC blocks, {len(summary_data)} summary stages.")

    plot_trend(log_entries, metric="elapsed")
    plot_trend(log_entries, metric="peak")
    plot_stage_pies(log_entries)
    print("Generated runtime_vs_nseg.png, peak_memory_vs_nseg.png, stage_breakdown.png")


if __name__ == "__main__":
    main()
