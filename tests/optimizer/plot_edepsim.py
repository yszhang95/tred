#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors


DATA_PATH = Path(__file__).with_name("lifetime_fit_results.npz")


def main():
    data = np.load(DATA_PATH)

    # loss = data["total_losses"]
    # lifetime = data["lifetime_values"]
    edepsim = data["edepsim"]

    # Ensure segments are shaped (N, >=8)
    if edepsim.ndim != 2 or edepsim.shape[1] < 8:
        raise ValueError(
            f"edepsim has unexpected shape {edepsim.shape}; "
            "expected (N_segments, >=8) with columns for charge, X0, X1."
        )

    x0 = edepsim[:, 2:5]  # starting points
    x1 = edepsim[:, 5:8]  # ending points
    dEdx = edepsim[:, 1]

    fig = plt.figure(figsize=(13, 9))

    # # Loss vs epoch
    # ax_loss = fig.add_subplot(2, 2, 1)
    # ax_loss.plot(epochs, loss, color="tab:blue")
    # ax_loss.set_title("Loss vs Epoch")
    # ax_loss.set_xlabel("Epoch")
    # ax_loss.set_ylabel("Loss")
    # ax_loss.grid(True, alpha=0.3)

    # # Lifetime vs epoch
    # ax_life = fig.add_subplot(2, 2, 2)
    # ax_life.plot(epochs, lifetime, color="tab:orange")
    # ax_life.set_title("Lifetime vs Epoch")
    # ax_life.set_xlabel("Epoch")
    # ax_life.set_ylabel("Lifetime")
    # ax_life.grid(True, alpha=0.3)

    # Segment projections
    projections = [
        ("X-Y Projection", 0, 1),
        ("X-Z Projection", 0, 2),
        ("Y-Z Projection", 1, 2),
    ]

    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=np.min(dEdx), vmax=np.max(dEdx))

    for idx, (title, dim_a, dim_b) in enumerate(projections, start=1):
        ax = fig.add_subplot(2, 2, idx)
        for start, end, q in zip(x0, x1, dEdx):
            ax.plot(
                [start[dim_a], end[dim_a]],
                [start[dim_b], end[dim_b]],
                color=cmap(norm(q)),
                marker="o",
                linewidth=1.5,
                markersize=3,
            )
        ax.set_title(title)
        labels = ["X", "Y", "Z"]
        ax.set_xlabel(labels[dim_a])
        ax.set_ylabel(labels[dim_b])
        ax.grid(True, alpha=0.2)

    # Shared colorbar for segment plots
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[fig.axes[i] for i in range(0, 3)], shrink=0.85)
    cbar.set_label("Segment dE/dx")

    fig.savefig("edepsim_plots.png", dpi=300)

if __name__ == "__main__":
    main()
