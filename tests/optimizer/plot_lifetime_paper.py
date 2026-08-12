#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


data = np.load("lifetime_fit_results.npz")
loss = data["total_losses"]
lifetime = data["lifetime_values"]
lifetime_inc = float(data["lifetime_inc"])
lifetime_dec = float(data["lifetime_dec"])

n_epochs = loss.size
midpoint = n_epochs // 2

loss_inc, loss_dec = loss[:midpoint], loss[midpoint:]
lifetime_inc_vals = lifetime[:midpoint]
lifetime_dec_vals = lifetime[midpoint:]

epochs_inc = np.arange(loss_inc.size)
epochs_dec = np.arange(loss_dec.size)

fig, ax = plt.subplots(1, 1, figsize=(7, 5.25))

ax.plot(epochs_inc, lifetime_inc_vals, label=f"Initial lifetime = True × {lifetime_inc:.2f}", color="tab:orange", linewidth=2.5)
ax.plot(epochs_dec, lifetime_dec_vals, label=f"Initial lifetime = True × {lifetime_dec:.2f}", color="tab:red", linewidth=2.5)
ax.axhline(0.8, color="k", linestyle="--", linewidth=1, label="True lifetime")
ax.set_xlabel("Epoch", fontsize=20)
ax.set_ylabel("Lifetime [ms]", fontsize=20)
# ax.set_title("Lifetime vs Epoch")
ax.grid(False)
ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=18)
ax.legend(fontsize=18, frameon=False)

fig.tight_layout()

# fig.savefig("lifetime_fit_results_paper.png", dpi=300)
for ext in ("png", "pdf"):
    fig.savefig(f"lifetime_fit_results_new.{ext}", dpi=300)
