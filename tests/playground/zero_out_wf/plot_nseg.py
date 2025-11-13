#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

log_path = Path("full.log")
values = [int(m.group(1)) for m in re.finditer(r"Nseg(\d+)", log_path.read_text())]
if not values:
    raise SystemExit("No Nseg entries found in full.log")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(values, bins=160, color="#1f77b4", edgecolor="black", alpha=0.75)
ax.set_xlabel("Nseg")
ax.set_ylabel("Count")
ax.set_title("Histogram of Nseg from full.log")

# Zoomed inset for 0–10k
zoom_ax = inset_axes(ax, width="42%", height="42%", loc="upper right")
zoom_ax.hist(values, bins=160, color="#ff7f0e", edgecolor="black", alpha=0.8)
zoom_ax.set_xlim(0, 10_000)
zoom_ax.set_xticks([0, 5_000, 10_000])
zoom_ax.set_yticklabels([])
zoom_ax.set_title("0 ≤ Nseg ≤ 10k", fontsize=9)

plt.tight_layout()
# plt.show()
plt.savefig('hist_nseg.png')
