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

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(epochs_inc, loss_inc, label=f"loss (×{lifetime_inc:.3f})", color="tab:blue")
axs[0].plot(epochs_dec, loss_dec, label=f"loss (×{lifetime_dec:.3f})", color="tab:green")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].set_title("Loss vs Epoch")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(epochs_inc, lifetime_inc_vals, label=f"lifetime (×{lifetime_inc:.3f})", color="tab:orange")
axs[1].plot(epochs_dec, lifetime_dec_vals, label=f"lifetime (×{lifetime_dec:.3f})", color="tab:red")
axs[1].axhline(0.8, color="k", linestyle="--", linewidth=1, label="reference 0.8")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Lifetime")
axs[1].set_title("Lifetime vs Epoch")
axs[1].grid(True)
axs[1].legend()

fig.tight_layout()

fig.savefig("lifetime_fit_results.png", dpi=300)
