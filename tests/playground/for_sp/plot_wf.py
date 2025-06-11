#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

f = np.load("/nfs/data/1/yousen/signal_processing/single_track_for_sp_unipolar.npz")

m = f['current_tpc2_batch11_location'][:,[0,1]] == f['hits_tpc2_batch11_location'][0][None,[0,1]]
m = m.all(axis=1)

current = np.squeeze(f['current_tpc2_batch11'][m])
cl = np.squeeze(f['current_tpc2_batch11_location'][m])[2]
hit =  f['hits_tpc2_batch11'][0]
hl =  np.squeeze(f['hits_tpc2_batch11_location'][0])[2]

thres = f['threshold']

x = np.arange(0, len(current))+cl
plt.plot(x, np.cumsum(current))

plt.vlines([hl,], 0, hit[-1])
print(hl, hit[-1])
plt.hlines(thres, x[0], x[-1], linestyle='--')
print(thres)

plt.xlim(1520, 1650)

plt.savefig("wf_example.png")
