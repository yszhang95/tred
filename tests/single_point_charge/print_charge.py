#!/usr/bin/env python3
import numpy as np
f = np.load("output.npz")
print(f['hits_tpc0_batch0'])
print(f.files)
print(np.sum(np.squeeze(f['current_tpc0_batch0']), axis=-1))
