#!/usr/bin/env python
from tred.graph import Drifter, Raster, ChunkSum, LacedConvo, Charge, Current, Sim
from tred.blocking import Block
from tred import units
from .response import get_ndlarsim
from tred.util import debug, info, tenstr
from tred.loaders import StepLoader, steps_from_ndh5
from tred.io_nd import (
    nd_collate_fn, create_tpc_datasets_from_steps,
    LazyLabelBatchSampler, EagerLabelBatchSampler, SortedLabelBatchSampler, CustomNDLoader, simple_geo_parser
)
from tred.recombination import birks, box
from tred.io import write_npz

import sys
import h5py
import numpy as np
import yaml
from collections import defaultdict
import torch
import time

import torch
import time

def make_nd(device='cpu'):
    '''
    This mocks up some file of depo sets.
    '''

    borders = simple_geo_parser('tests/nd_geometry/ndlar-module.yaml', 'tests/nd_geometry/multi_tile_layout-3.0.40.yaml')
    # path = '/home/yousen/Public/ndlar_shared/data/segments_a_muon_first_event.hdf5'
    path = '/home/yousen/Public/ndlar_shared/data/segments_first_event.hdf5'
    d0 = StepLoader(h5py.File(path), transform=steps_from_ndh5)
    f0, f1, i0 = d0[:]
    return (f0, f1, i0), i0, borders


def segment_to_tpc(features, labels, borders):
    tpcs = create_tpc_datasets_from_steps(features, labels, borders, sort_index=1)
    return tpcs

def runit(device='cpu'):
    '''
    '''

    # eventually replace this hard-wire with configuration
    DL = 7.2 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    DT = 12.0 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    diffusion = torch.tensor([DL, DT, DT])
    lifetime = 8*units.ms / units.us # values are in units of us
    velocity = 1.6 * units.mm/units.us / (units.cm/units.us) # values are in units of cm/us
    pitch = 4.4*units.mm / units.cm # values are in units of cm
    nimperpix=10
    pspace = pitch/nimperpix
    tspace = 50*units.ns / units.us # values are in units of us
    grid_spacing = (pspace, pspace, tspace)
    # npixpersuper = 10
    npixpersuper = 2
    ntickperslice = 300
    chunk_shape = (npixpersuper * nimperpix, npixpersuper * nimperpix, ntickperslice)
    print(chunk_shape)

    efield = 0.5 # kV/cm
    rho = 1.38 # g/cm^3
    A3t = 0.8 # birks
    k3t = 0.0486 # (g/MeV cm^2) (kV/cm); birks
    Wi = 23.6E-6 # MeV/pair

    lacing = torch.tensor([nimperpix, nimperpix, 1])

    batch_size = 4096

    t0 = time.time()
    # create intermediate nodes
    # dummy drifter for running time tests
    drifter = Drifter(diffusion, lifetime, velocity)
    raster = Raster(velocity, grid_spacing)
    chunksum = ChunkSum(chunk_shape)
    convo = LacedConvo(lacing)

    t1 = time.time()

    response = get_ndlarsim()

    response = response.to(device=device)

    t2 = time.time()

    tpcs = segment_to_tpc(*make_nd('cpu'))

    t3 = time.time()

    runtime = defaultdict(list)

    waveforms = {}

    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


    for itpc, tpcdataset in enumerate(tpcs):
        # only look at the TPC 26 for now for the test ...
        if (itpc!=26) : continue

        sampler = SortedLabelBatchSampler(tpcdataset.labels[:,0], batch_size)
        loader = CustomNDLoader(tpcdataset, sampler=sampler,
                                batch_size=None, collate_fn=nd_collate_fn)
        drifter = Drifter(diffusion, lifetime, tpcdataset.drift*velocity,
                          target=tpcdataset.anode)
        drifter = drifter.to(device=device)

        raster = raster.to(device=device)
        chunksum = chunksum.to(device=device)
        convo = convo.to(device=device)

        tpc_lower_left = tpcdataset.lower_left_corner.to(device).unsqueeze(0)

        for ibatch, (features, labels) in enumerate(loader):
            stime = time.time()
            try:
                if device == 'cuda':
                    torch.cuda.synchronize()
                t00 = time.time()
                features = [f.to(device=device) for f in features]
                labels = labels.to(device=device)

                if device == 'cuda':
                    torch.cuda.synchronize()
                t01 = time.time()

                charge = birks(dE=features[0][:,0], dEdx=features[0][:,1],
                          efield=efield, rho=rho, A3t=A3t, k3t=k3t, Wi=Wi)
                if device == 'cuda':
                    torch.cuda.synchronize()
                t02 = time.time()

                local_time = features[0][:,-1]
                tail = features[0][:,2:5]
                head = features[0][:,5:8]
                tail[:,[1,2]] -= tpc_lower_left
                head[:,[1,2]] -= tpc_lower_left

                # dsigma, dtime, dcharge, dtail, dhead
                drifted = drifter(local_time, charge, tail, head)

                if device == 'cuda':
                    torch.cuda.synchronize()
                t03 = time.time()

                qblock = raster(*drifted)

                # print('drifted', 'sigma>1E-2', torch.all(drifted[0]>1E-2), torch.all(drifted[1]>0), 'input charge >1000', torch.all(drifted[2]>1000))
                # print('sum of qblock', torch.sum(qblock.data))
                # print('drifted', 'sigma', drifted[0][:2],
                #       'time', drifted[1][:2], 'charge', drifted[2][:2],
                #       'tail', drifted[3][:2], 'head', drifted[4][:2])

                if device == 'cuda':
                    torch.cuda.synchronize()
                t04 = time.time()

                signal = chunksum(qblock)
                qblock = None

                if device == 'cuda':
                    torch.cuda.synchronize()
                t05 = time.time()

                # free_mem, total_mem = torch.cuda.mem_get_info()
                # total_mem = torch.cuda.get_device_properties(0).total_memory
                # # r = torch.cuda.memory_reserved(0)
                # allocated = torch.cuda.memory_allocated(0)
                # free_mem = total_mem - allocated
                # print('mem snapshot', free_mem/1024**3, total_mem/1024**3)

                # if free_mem/total_mem > 0.7:
                #     print(free_mem/total_mem)
                #     nchunks = 1
                # elif free_mem/total_mem > 0.5:
                #     nchunks = 3
                # elif free_mem/total_mem > 0.3:
                #     nchunks = 6
                # elif free_mem/total_mem > 0.2:
                #     nchunks = 9
                # else:
                #     nchunks = 15
                # print(nchunks)
                nchunks = signal.nbatches // 300 + 1
                ibs = []
                for l, s in zip(signal.location.chunk(nchunks), signal.data.chunk(nchunks)):
                    # print(s.shape[0])
                    if s.shape[0] > 0:
                        iblock = convo(Block(location=l, data=s), response)

                if device == 'cuda':
                    torch.cuda.synchronize()
                t06 = time.time()

                # current =chunksum(iblock)

                if device == 'cuda':
                    torch.cuda.synchronize()
                t07 = time.time()

                runtime['to_device'].append(t01-t00)
                runtime['recomb'].append(t02-t01)
                runtime['drift'].append(t03-t02)
                runtime['raster'].append(t04-t03)
                runtime['chunksum_charge'].append(t05-t04)
                runtime['convo'].append(t06-t05)
                runtime['chunksum_current'].append(t07-t06)

                info(f'{runtime["to_device"][-1]} data to {device}')
                info(f'{runtime["recomb"][-1]} recomb')
                info(f'{runtime["drift"][-1]} drift')
                info(f'{runtime["raster"][-1]} raster')
                info(f'{runtime["chunksum_charge"][-1]} chunksum_charge')
                info(f'{runtime["convo"][-1]} convo')
                info(f'{runtime["chunksum_current"][-1]} chunksum_current')

                if device == 'cuda':
                    cuda_mem = torch.cuda.max_memory_allocated() / 1024**2
                    info(f'Peak cuda usage: {cuda_mem} MB')

                info(f'itpc{itpc}, tpc label {tpcdataset.tpc_id}, batch label {ibatch}, '
                      f'N segments {len(features[0])}, '
                      f'elapsed {t07 - stime} sec on {device}.')

                # waveforms[f'current_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = current
                # waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = signal

                torch.cuda.reset_peak_memory_stats()
            except IndexError:
                pass
            # except torch.OutOfMemoryError as e:
            #     current_device = torch.cuda.current_device()
            #     allocated = torch.cuda.memory_allocated(current_device) / (1024 ** 2)  # in MB
            #     reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 2)    # in MB
            #     total = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 2)  # in MB

            #     info('torch.OutOfMemoryError, '
            #         f'tpc label {itpc}, batch label {ibatch}, '
            #         f'N segments {len(features[0])}, '
            #         f'elapsed {time.time() - stime} sec on {device}. '
            #         f'Memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB, '
            #         f'total: {total:.2f} MB.'
            #         f' {e}')

    try:
        torch.cuda.memory._dump_snapshot(f"crop_batched.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")

    # Stop recording memory snapshot history.
    torch.cuda.memory._record_memory_history(enabled=None)

    write_npz("waveforms.npz", **waveforms)

    info(f'{t1-t0} construct')
    info(f'{t2-t1} get response')
    info(f'{t3-t2} load nd from disk')
    info(f'{sum(runtime["to_device"])} data to {device}')
    info(f'{sum(runtime["recomb"])} recomb')
    info(f'{sum(runtime["drift"])} drift')
    info(f'{sum(runtime["raster"])} raster')
    info(f'{sum(runtime["chunksum_charge"])} chunksum_charge')
    info(f'{sum(runtime["convo"])} convo')
    info(f'{sum(runtime["chunksum_current"])} chunksum_current')

    info(f'Total elapsed time {time.time() - t0} seconds')

def plots(out):
    with torch.no_grad():
        # torch.set_default_device('cuda')
        # runit('cpu')
        # info('FINISHED CPU')
        runit('cuda')
        info('FINISHED CUDA')
