#!/usr/bin/env python3

from tred.toy_gpu3 import QEff3D, UniversalGrid, LocalGrid

from tred.drift import drift

from tred.loaders import StepLoader

from tred.recombination import birks

import numpy as np

import sys
import time
import h5py
import torch

f = '/home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5'
# f = '/home/yousen/Public/ndlar_shared/data/segments_first_event.hdf5'

#batch_size = 10_000
batch_size = 2_500
# batch_size = 100
efield = 0.5 # kV/cm
rho = 1.38 # g/cm^3
A3t = 0.8 # birks
k3t = 0.0486 # (g/MeV cm^2) (kV/cm); birks
Wi = 23.6E-6 # MeV/pair

origin = torch.tensor((0.,0.,0.), requires_grad=False, device='cuda:0')
# grid_spacing = (0.0443, 0.0443, 0.016)
grid_spacing = (0.016, 0.0443, 0.0443)

n_sigma = torch.tensor((3., 3., 3.), requires_grad=False, device='cuda:0')

def run_one_event(dataset):
    steps = StepLoader(dataset)

    nsteps = len(steps)
    nbatches = nsteps // batch_size + 1


    charges = []

    counter = 0

    torch.cuda.synchronize()
    start_time = time.time()

    for i in range(0, nsteps, batch_size):
        charge_batches = []
        # idxs = slice(i, min(nsteps, i+batch_size))
        idxs = slice(i, i+batch_size)
        batched_data = steps[idxs]
        segments = batched_data[0].to('cuda:0')

        Q = birks(dE=segments[:,0], dEdx=segments[:,1], efield=efield, rho=rho, A3t=A3t, k3t=k3t, Wi=Wi)
        X0 = segments[:,2:5]
        X1 = segments[:,5:8]
        Sigma = 0.2*torch.ones((len(Q), 3), requires_grad=False, device='cuda:0')
        dX = X1 - X0

        nsegments = len(Q)
        subbatch_size = 100
        for j in range(0, nsegments, subbatch_size):
            # segidxs = slice(j, min(nsegments, j+subbatch_size))
            segidxs = slice(j, j+subbatch_size)
            Qi = Q[segidxs]
            X0i = X0[segidxs]
            X1i = X1[segidxs]
            Sigmai = Sigma[segidxs]
            offset, shape = LocalGrid.compute_charge_box(X0=X0i, X1=X1i, Sigma=Sigmai,
                                                         n_sigma=n_sigma, origin=origin, grid_spacing=grid_spacing)
            # 200000 > 192000 = 40*40*120
            if shape[0]*shape[1]*shape[2] > 200000:
                subsubbatch_size = 20
                nsubsegments = len(Qi)
                for k in range(0, nsubsegments, subsubbatch_size):
                    # kidxs = slice(k, min(nsubsegments, k+subsubbatch_size))
                    kidxs = slice(k, k+subsubbatch_size)
                    Qii = Qi[kidxs]
                    X0ii = X0i[kidxs]
                    X1ii = X1i[kidxs]
                    Sigmaii = Sigmai[kidxs]
                    offset, shape = LocalGrid.compute_charge_box(X0=X0ii, X1=X1ii, Sigma=Sigmaii,
                                                                n_sigma=n_sigma, origin=origin, grid_spacing=grid_spacing)
                    if shape[0]*shape[1]*shape[2] > 200000:
                        nsubsubsegments = len(Qii)
                        subsubsubbatch_size = 2
                        for l in range(0, nsubsubsegments, subsubsubbatch_size):
                            lidxs = slice(l, l+subsubsubbatch_size)
                            Qiii = Qii[lidxs]
                            X0iii = X0ii[lidxs]
                            X1iii = X1ii[lidxs]
                            Sigmaiii = Sigmaii[lidxs]
                            offset, shape = LocalGrid.compute_charge_box(X0=X0iii, X1=X1iii, Sigma=Sigmaiii,
                                                                n_sigma=n_sigma, origin=origin, grid_spacing=grid_spacing)
                            qeff = QEff3D.eval_qeff(Qiii, X0iii, X1iii, Sigmaiii, offset, shape, origin, grid_spacing, method='gauss_legendre', npoints=(2,2,2))
                            charge_batches.append(qeff.to('cpu'))
                    else:
                        qeff = QEff3D.eval_qeff(Qii, X0ii, X1ii, Sigmaii, offset, shape, origin, grid_spacing, method='gauss_legendre', npoints=(2,2,2))
                        charge_batches.append(qeff.to('cpu'))
            else:
                qeff = QEff3D.eval_qeff(Qi, X0i, X1i, Sigmai, offset, shape, origin, grid_spacing, method='gauss_legendre', npoints=(2,2,2))
                charge_batches.append(qeff.to('cpu'))
            torch.cuda.empty_cache()

        charges.append(charge_batches)

    torch.cuda.synchronize()
    end_time = time.time()

    dt = end_time - start_time

    memGPU = torch.cuda.max_memory_allocated()/ 1024**2
    torch.cuda.reset_peak_memory_stats()

    memCPUs = []
    for i, cbs in enumerate(charges):
        memCPUs.append(0)
        nel = 0
        for c in cbs:
            nel += c.numel()
        memCPUs[i] = 4* nel /1024**2
    memCPU = np.max(memCPUs)

    print(f'Elapsed time {dt:.3f}sec')
    print(f'Peak GPU memory usage {memGPU:.0f} MB')
    print(f'Peak CPU memory {memCPU:.0f} MB')

    return dt, memGPU, memCPU


def main():
    fhdf5 = h5py.File(f, 'r')
    eventids = np.unique(fhdf5['segments']['event_id'])
    dts = []
    memGPUs = []
    memCPUs = []
    for i, eventid in enumerate(eventids):
        # if i<=50 or i>60:
        #     continue
        # if i>50:
        #     continue
        # if i <=60:
        #     continue
        segments = fhdf5['segments'][ fhdf5['segments']['event_id'] == eventid ]
        dt, memGPU, memCPU = run_one_event(segments)
        dts.append(dt)
        memGPUs.append(memGPU)
        memCPUs.append(memCPU)
        print('Finished', i, 'event')

    dts = np.array(dts)
    memGPUS = np.array(memGPUs)
    memCPUS = np.array(memCPUs)

    np.savez('profile_nd.npz', dts=dts, memGPUs=memGPUs, memCPUs=memCPUs)

if __name__ == '__main__':
    main()
