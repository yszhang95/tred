#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

f = np.load('output.npz')
adc_hold_delay = f['adc_hold_delay']
# nburst = f['nburst']
nburst = 8
curr_loc = f['current_tpc0_batch0_location']
curr = f['current_tpc0_batch0']
curr = np.squeeze(curr)
hits_loc = f['hits_tpc0_batch0_location']
hits = f['hits_tpc0_batch0']

nhits = len(hits)

fig, ax = plt.subplots(nhits, 1)
for ihit in range(nhits):
    hpixloc = hits_loc[ihit][:2]
    hloc = hits_loc[ihit]
    cpixmask = np.all(curr_loc[:, :2] == hpixloc, axis=-1)
    cloc = curr_loc[cpixmask]
    if curr[cpixmask].shape[0] !=1:
        raise ValueError(f"Wrong shape of selected current waveform {curr[cpixmask].shape}")
    print(hpixloc, cloc)
    cum_curr = np.cumsum(curr[cpixmask], axis=-1)
    print(cum_curr.shape)
    print(cloc[0, -1])
    ctimes = cloc[0, -1] + np.arange(cum_curr.shape[-1])
    ax[ihit].plot(ctimes, np.squeeze(cum_curr), label='cumulative charge')
    htimes = hloc[2] + adc_hold_delay + np.arange(nburst) * adc_hold_delay
    hqs = hits[ihit][3:]
    ax[ihit].plot(htimes, hqs, 'o', label='burst hits')
    ax[ihit].legend()

    ax[ihit].set_xlim(ctimes[0], htimes[-1] + 3*adc_hold_delay)

fig.savefig("burst_charge_records.png")
