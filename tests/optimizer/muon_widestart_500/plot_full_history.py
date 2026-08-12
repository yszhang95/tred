#!/usr/bin/env python3
'''
Plot the concatenated (original + dec-extension) history of the 500 e-/tick
wide-start lifetime fit: tuned lifetime and total loss vs epoch.
Reads lifetime_fit_results_full.npz written by concat_history.py.
'''
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(here, "lifetime_fit_results_full.npz"))

inc_lt, inc_ls = data["inc_lifetimes"], data["inc_losses"]
dec_lt, dec_ls = data["dec_lifetimes"], data["dec_losses"]
lifetime_inc = float(data["lifetime_inc"])
lifetime_dec = float(data["lifetime_dec"])
n_dec_orig = int(data["n_dec_orig"])

fig, (ax, axl) = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2]})

ax.plot(np.arange(inc_lt.size), inc_lt,
        label=f"Initial lifetime = True × {lifetime_inc:.2f}",
        color="tab:orange", linewidth=2.5)
ax.plot(np.arange(dec_lt.size), dec_lt,
        label=f"Initial lifetime = True × {lifetime_dec:.2f}",
        color="tab:red", linewidth=2.5)
ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="True lifetime")
ax.set_ylabel("Lifetime [ms]", fontsize=20)
ax.grid(False)
ax.tick_params(axis="both", which="both", direction="in", top=True, right=True,
               labelsize=18)
ax.legend(fontsize=16, frameon=False)

# the MSE sits a fraction of a percent above the sigma^2 noise floor; show
# the excess above the best achieved loss so convergence is visible
L_ref = min(inc_ls.min(), dec_ls.min())
eps = np.min([d[d > 0].min() for d in (inc_ls - L_ref, dec_ls - L_ref)
              if (d > 0).any()]) / 2
axl.plot(np.arange(inc_ls.size), np.clip(inc_ls - L_ref, eps, None),
         color="tab:orange", linewidth=2.5)
axl.plot(np.arange(dec_ls.size), np.clip(dec_ls - L_ref, eps, None),
         color="tab:red", linewidth=2.5)
axl.set_xlabel("Epoch", fontsize=20)
axl.set_ylabel(r"Loss $-$ $L_{\mathrm{min}}$", fontsize=20)
axl.set_yscale("log")
axl.grid(False)
axl.tick_params(axis="both", which="both", direction="in", top=True, right=True,
                labelsize=18)

fig.tight_layout()
for ext in ("png", "pdf"):
    out = os.path.join(here, f"lifetime_fit_500e_full.{ext}")
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
print(f"inc final: {inc_lt[-1]:.6f} ms, dec final: {dec_lt[-1]:.6f} ms, "
      f"diff {(inc_lt[-1]-dec_lt[-1])*1e3:.3f} us")
