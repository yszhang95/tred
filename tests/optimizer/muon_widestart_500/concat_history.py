#!/usr/bin/env python
'''
Concatenate the wide-start 500 e- run history with its dec-arm extension.

Original run: [inc 2600 epochs][dec 2600 epochs], starts 2.0 / 0.5 ms.
Extension:    [inc 1 epoch (eval only)][dec 1200 epochs], dec resumed at the
              original dec arm's final value with the same anchored noise.

Writes lifetime_fit_results_full.npz with per-arm concatenated histories.
The extension's dec epoch 0 re-evaluates the resume point (same value as the
original's last entry) and is dropped to avoid a duplicate sample.
'''
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
orig = np.load(os.path.join(here, "lifetime_fit_results.npz"))
ext = np.load(os.path.join(here, "dec_extension", "lifetime_fit_results.npz"))

n_inc, n_dec = 2600, 2600
n_inc_ext = 1

lt_o, ls_o = orig["lifetime_values"], orig["total_losses"]
lt_e, ls_e = ext["lifetime_values"], ext["total_losses"]
assert len(lt_o) == n_inc + n_dec, len(lt_o)

inc_lt = lt_o[:n_inc]
inc_ls = ls_o[:n_inc]
dec_lt = np.concatenate([lt_o[n_inc:], lt_e[n_inc_ext + 1:]])  # drop ext dec epoch 0
dec_ls = np.concatenate([ls_o[n_inc:], ls_e[n_inc_ext + 1:]])

out = os.path.join(here, "lifetime_fit_results_full.npz")
np.savez(out,
         inc_lifetimes=inc_lt, inc_losses=inc_ls,
         dec_lifetimes=dec_lt, dec_losses=dec_ls,
         lifetime_values=np.concatenate([inc_lt, dec_lt]),
         total_losses=np.concatenate([inc_ls, dec_ls]),
         n_inc=len(inc_lt), n_dec_orig=n_dec, n_dec_ext=len(lt_e) - n_inc_ext - 1,
         lifetime_inc=float(orig["lifetime_inc"]),
         lifetime_dec=float(orig["lifetime_dec"]))
print(f"wrote {out}")
print(f"inc arm: {len(inc_lt)} epochs, final {inc_lt[-1]:.6f} ms")
print(f"dec arm: {len(dec_lt)} epochs (2600 + extension), final {dec_lt[-1]:.6f} ms")
print(f"arm difference: {(inc_lt[-1]-dec_lt[-1])*1e3:.4f} us")
