#!/usr/bin/env python3
"""
Plot grouped current waveforms (true/inc/dec) from lifetime_fit_results.npz.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DATA_PATH = Path(__file__).with_name("lifetime_fit_results.npz")
OUT_PATH = Path(__file__).with_name("current_groups.pdf")
TIME_TICK = 1.0          # adjust if you know the actual tick spacing
GROUP_THRESHOLD = 1.0
OFFSET_GRID = [(dx, dy) for dy in (1, 0, -1) for dx in (-1, 0, 1)]


def load_data():
    data = np.load(DATA_PATH)
    return (
        data["currents_data_true"],
        data["currents_data_inc_0"],
        data["currents_data_dec_0"],
        data["currents_location_true"],
    )


def flatten_waveforms(block):
    """(N, 1, 1, T) -> (N, T)."""
    return block.reshape(block.shape[0], -1)


def find_group_centers(location, waveforms, threshold):
    sums = waveforms.sum(axis=-1)
    mask = sums > threshold
    return location[mask, :2]


def indices_by_offset(location, center):
    coords = location[:, :2]
    center = np.asarray(center)
    result = {}
    for dx, dy in OFFSET_GRID:
        target = center + np.array([dx, dy])
        hits = np.nonzero(np.all(coords == target, axis=1))[0]
        result[(dx, dy)] = hits
    return result


def plot_groups(true_wf, inc_wf, dec_wf, location, centers):
    legend_handles = [
        plt.Line2D([], [], color="tab:blue", linestyle='-', label="true"),
        plt.Line2D([], [], color="tab:orange", linestyle='-.', label="inc"),
        plt.Line2D([], [], color="tab:green", linestyle='--', label="dec"),
    ]

    with PdfPages(OUT_PATH) as pdf:
        for center in centers:
            offset_indices = indices_by_offset(location, center)

            if all(idx.size == 0 for idx in offset_indices.values()):
                continue

            fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=False, sharey=False)
            fig.suptitle(f"Pixel group around ({int(center[0])}, {int(center[1])})")

            for ax, (dx, dy) in zip(axes.flat, OFFSET_GRID):
                idx = offset_indices[(dx, dy)]
                pixel_label = (int(center[0] + dx), int(center[1] + dy))

                if idx.size == 0:
                    ax.set_axis_off()
                    ax.set_title(f"{pixel_label} (empty)", fontsize=10)
                    continue

                for i in idx:
                    t0 = location[i, 2]
                    times = t0 + np.arange(true_wf.shape[1]) * TIME_TICK
                    ax.plot(times, true_wf[i], color="tab:blue", linestyle='-', alpha=0.6)
                    ax.plot(times, inc_wf[i], color="tab:orange", linestyle='-.', alpha=0.6)
                    ax.plot(times, dec_wf[i], color="tab:green", linestyle='--', alpha=0.6)
                    ax.set_xlim(0, 2500)

                ax.set_title(f"{pixel_label}", fontsize=10)
                ax.grid(True, alpha=0.3)

            for ax in axes[-1, :]:
                ax.set_xlabel("Time (ticks)")
            for ax in axes[:, 0]:
                ax.set_ylabel("Current")

            fig.legend(handles=legend_handles, loc="upper right")
            fig.tight_layout(rect=[0, 0, 0.95, 0.96])

            pdf.savefig(fig)
            plt.close(fig)


def main():
    true_block, inc_block, dec_block, location = load_data()
    true_wf = flatten_waveforms(true_block)
    inc_wf = flatten_waveforms(inc_block)
    dec_wf = flatten_waveforms(dec_block)

    centers = find_group_centers(location, true_wf, GROUP_THRESHOLD)
    plot_groups(true_wf, inc_wf, dec_wf, location, centers)


if __name__ == "__main__":
    main()
