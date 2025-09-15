#!/usr/bin/env python3
import numpy as np

fres = np.load("/home/yousen/Public/ndlar_shared/data/responses/response_5x5_shield_900V_pitch0p372cm_tick0p1us_drtoa30p431cm.npz")
fout = {
}

for f in fres.files:
    fout[f] = fres[f]
    if f == "bin_size":
        fout[f] = 0.04434  # fake to v2a

np.savez_compressed("/home/yousen/Public/ndlar_shared/data/responses/response_5x5_shield_900V_pitch0p372cm_fakepitch0p4434cm_tick0p1us_drtoa30p431cm_v2a.npz", **fout)
