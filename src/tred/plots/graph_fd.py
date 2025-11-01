#!/usr/bin/env python
from tred.graph import Drifter, Raster, ChunkSum, LacedConvo, Charge, Current, Sim
from tred.response import ndlarsim
from tred.blocking import Block, concat_blocks, iter_chunk_block
from tred import units
from .response import get_ndlarsim
from tred.util import debug, info, tenstr, warning, log, iter_tensor_chunks
from tred.loaders import StepLoader, steps_from_ndh5
from tred.io_nd import (
    nd_collate_fn, create_tpc_datasets_from_steps,
    LazyLabelBatchSampler, EagerLabelBatchSampler, SortedLabelBatchSampler, CustomNDLoader, simple_geo_parser
)
from tred.recombination import birks, box
from tred.io import write_npz
from tred import chunking
from tred.readout import nd_readout

import sys
import h5py
import numpy as np
import yaml
from collections import defaultdict
import torch
import time
import os

import torch
import torch.nn as nn
import time

response_path = None
lifetime = None
input_path = None
output_path = None
drtoa = None
tspace = None
event_list = None
save_waveform = None

tspace_out = 0.5 * units.us / units.us
pitch = 4.8 * units.mm / units.cm # values are in units of cm
nimperpix=10
pspace = pitch/nimperpix
velocity = 1.59645 * units.mm/units.us / (units.cm/units.us) # values are in units of cm/us
pspace_rot = 0.47 * units.mm / units.cm

response = None

def concatenate_waveforms(sparse_currents, Nt, event_t=0):
    '''
     Assume there is no overlap.
     Assume location is binned into 1x1 pixel groups.
     location: shape (Nbatch, vdim)
     data: shape (Nbatch, 1, 1, ,1, ..., Mt)
     Nt is the length of output along time axis (last axis). It must be divisible by Mt.
     '''
    data = sparse_currents.data
    location = sparse_currents.location
    Mt = data.shape[-1]
    if Nt % Mt:
        raise ValueError(f'Nt: {Nt} must be divisible by Mt: {Mt}')
    if any(l != 1 for l in data.shape[1:-1]):
        raise ValueError(f'Size in pixel domains must be 1, but {data.shape[1:-1]} is given.')

    vdim = location.shape[1]
    pixel_locs, rev_ind = torch.unique(location[:, :-1], dim=0, return_inverse=True, sorted=True)
    Npix = pixel_locs.size(0)
    Nbatch = location.shape[0]

    # Construct loc_out = [pixel_coords..., min_time_per_pixel]
    min_time = torch.full((Npix,), Nt+torch.max(location[:,-1]), device=location.device, dtype=location.dtype)
    min_time.scatter_reduce_(0, rev_ind, location[:, -1], reduce='amin', include_self=False)
    loc_out = torch.cat([pixel_locs, min_time.unsqueeze(1)], dim=1)  # shape (Npix, vdim)

    tref = min_time[rev_ind]

    # Initialize output waveform
    wf_out = torch.zeros((Npix, Nt), device=data.device, dtype=data.dtype)

    # Time assignment
    t_start = location[:, -1] - tref
    if torch.any(t_start<0):
        raise ValueError
    time_offsets = torch.arange(Mt, device=data.device).view(1, -1)
    time_indices = t_start.view(-1, 1) + time_offsets
    batch_indices = rev_ind.view(-1, 1).expand(-1, Mt).clone().detach()
    flat_batch = batch_indices.reshape(-1)
    flat_time = time_indices.reshape(-1)
    flat_values = data.view(-1, Mt).reshape(-1)
    wf_out.index_put_((flat_batch, flat_time), flat_values, accumulate=False)


    # Reshape waveform output to match original spatial dims
    wf_out = wf_out.view(Npix, *data.shape[1:-1], Nt)

    # filter negative ticks
    # Zero out any samples before event_t
    # For each pixel, if its start-time < event_t, zero samples index < event_t - start_time
    global_start = loc_out[:, -1]
    offsets = (event_t - global_start).clamp(min=0).to(torch.long)
    T = wf_out.size(-1)
    # mask_i,t = True if t < offsets[i]
    mask = (torch.arange(T, device=wf_out.device)[None, :] < offsets.to(wf_out.device)[:, None])
    # shape (Npix, T) -> insert singleton dims to cover the "..." in wf_out
    mask = mask.view(offsets.size(0), *([1] * (wf_out.ndim - 2)), T)
    # zero in-place where mask is True
    wf_out.masked_fill_(mask, 0)

    return Block(data=wf_out, location=loc_out)


def make_nd(device='cpu'):
    '''
    This mocks up some file of depo sets.
    '''

    borders = simple_geo_parser(module_yaml, tile_yaml, old_geo_config)
    d0 = StepLoader(h5py.File(input_path), transform=steps_from_ndh5)
    f0, f1, i0 = d0[:]
    return (f0, f1, i0), i0, borders


def segment_to_tpc(features, labels, borders):
    tpcs = create_tpc_datasets_from_steps(features, labels, borders, sort_index=0)
    return tpcs


def transform_indices_to_coord_3d(location, pitch, tick, velocity,
                                  lower, anode, direction,
                                  paxes=(0,1), taxis=-1, offset=None):
    '''
    location: batched
    pitch: in cm
    tick: in us
    velocity: in cm/us
    lower: in cm
    anode: in cm
    direction: -1 or +1
    '''
    if offset is None:
        offset = torch.zeros((1,3,), dtype=torch.int32, device=location.device)
    locs = location.to(torch.float32) + offset
    locs[:,paxes] = locs[:,paxes].to(torch.float32)*pitch + lower.to(locs.device)
    locs[:,taxis] = anode - direction * velocity * tick * locs[:,taxis].to(torch.float32)
    return locs

def runit(device='cpu'):
    '''
    '''
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    info(f'Starting runit on device {device}.')
    # everything is 2D
    export_pickle = False

    # eventually replace this hard-wire with configuration
    twindow_max = 2_000 # 2_000 * 0.50us = 1000us
    DL = 4.0 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    DT = 8.8 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    diffusion = torch.tensor([DL, DT])
    grid_spacing = (pspace, tspace)
    npixpersuper = 32+1-25
    ntickperslice = 64 # tspace == 0.5us
    chunk_shape = (npixpersuper * nimperpix, ntickperslice)

    efield = 0.5 # kV/cm
    rho = 1.38 # g/cm^3
    A3t = 0.8 # birks
    k3t = 0.0486 # (g/MeV cm^2) (kV/cm); birks
    Wi = 23.6E-6 # MeV/pair

    lacing = torch.tensor([nimperpix, 1])

    batch_size = 4096 * 8

    # create intermediate nodes
    chunksum = ChunkSum(chunk_shape)
    chunksum_readout = ChunkSum((1, 120))

    convo = LacedConvo(lacing, o_shape=(32, 512)) # tspace =0.5us
    chunksum_i = ChunkSum((16, 128), method='chunksum_inplace_v2')

    chunksum_i = chunksum_i.to(device)
    chunksum_readout = chunksum_readout.to(device)

    angle = np.deg2rad(35.7)
    vec_counterclock = np.array([np.cos(angle), np.sin(angle)])
    vec_counterclock = torch.tensor(vec_counterclock).to(torch.float32).to(device)
    vec_clock = np.array([np.cos(angle), -np.sin(angle)])
    vec_clock = torch.tensor(vec_clock).to(torch.float32).to(device)

    global response
    response = response.to(device=device)

    features = np.load(input_path, allow_pickle=True)

    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    # MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000
    if export_pickle:
        torch.cuda.memory._record_memory_history(
        # max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )

    # only one hypothetical TPC for now;
    # anode is at x=0, (positive drift for the given dataset)
    # lower corner = (0, )
    # let us assume (x, y, z) are mm and we want to convert things to cm
    features = features[features[:,0]>=0]
    features[:, 2:5] = features[:, 2:5] * units.mm / units.cm # to cm
    xmin = features[:,2].min()
    ymin = features[:,3].min()
    zmin = features[:,4].min()
    tpc_lower_corner = torch.tensor([xmin//pspace * pspace - 2,
                                     ymin//pspace * pspace - 2,
                                     zmin//pspace * pspace - 2], dtype=torch.float32)
    target = features[:,2].max() // pspace * pspace + pspace + drtoa
    drift_direction = 1.0
    drifter = Drifter(diffusion, lifetime, drift_direction*velocity, fluctuate=fluctuate,
                      target=target, drtoa=drtoa)
    drifter = drifter.to(device=device)

    raster = Raster(drift_direction*velocity, grid_spacing, pdims=(1,)).to(device=device)
    raster_rot = Raster(drift_direction*velocity, grid_spacing=(pspace_rot, tspace), pdims=(1,)).to(device=device)
    chunksum = chunksum.to(device=device)
    convo = convo.to(device=device)

    features = torch.tensor(features, dtype=torch.float32).to(device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    tstart = time.time()

    # rot_counterclock_tail = rot_counterclock @ features[:,None,3:5]
    # [N, 2] @ [2] --> [N}
    rot_counterclock_tail = torch.matmul(features[:,3:5],  vec_counterclock)
    tail_counterclock = torch.stack([features[:,2], rot_counterclock_tail], dim=1)
    rot_clock_tail = torch.matmul(features[:,3:5],  vec_clock)
    tail_clock = torch.stack([features[:,2], rot_clock_tail], dim=1)

    info(f"Start test run on {features.size(0)} segments.")
    batch_size = features.size(0)

    for i in range(0, features.size(0), batch_size):
        batch_features = features[i:i+batch_size]
        tail = batch_features[:,2:5]
        tail_p = tail_counterclock[i:i+batch_size]
        tail_n = tail_clock[i:i+batch_size]
        Q = batch_features[:,1] * (-1) # Nelectrons
        t0 = batch_features[:,0] * units.ns / units.us # us
        # global tref from min and offset for the rest
        tref = torch.min(t0)
        local_time = t0 - tref

        # X plane
        # drift and vertical
        drifted = drifter(local_time, Q, tail[:,:2])
        for ichunk, idrifted in enumerate(
                iter_tensor_chunks(drifted, chunk_size=batch_size)):
            qblock = raster(*idrifted)
            if fluctuate:
                qsum = qblock.data.sum(dim=(-1,-2), keepdim=True)
                variations = torch.randn_like(qblock.data)
                qblock.data += variations
                qblock.data /= qsum

            signal = chunksum(qblock)

            # Nqblock = signal.nbatches
            iblock = convo(signal, response)
            current = chunksum_i(iblock)
            readout_segmetns = chunksum_readout(current)
            concatenate_waveforms(readout_segmetns, Nt=19200)

        # + 35.7 deg plane; rotate on vertical and beam
        drifted = drifter(local_time, Q, tail_p[:,:2])
        for ichunk, idrifted in enumerate(
                iter_tensor_chunks(drifted, chunk_size=batch_size)):
            qblock = raster(*idrifted)
            if fluctuate:
                qsum = qblock.data.sum(dim=(-1,-2), keepdim=True)
                variations = torch.randn_like(qblock.data)
                qblock.data += variations
                qblock.data /= qsum

            signal = chunksum(qblock)

            # Nqblock = signal.nbatches
            iblock = convo(signal, response)
            current = chunksum_i(iblock)
            readout_segmetns = chunksum_readout(current)
            concatenate_waveforms(readout_segmetns, Nt=19200)

        #  - 35.7 deg plane;
        drifted = drifter(local_time, Q, tail_n[:,:2])
        for ichunk, idrifted in enumerate(
                iter_tensor_chunks(drifted, chunk_size=batch_size)):
            qblock = raster(*idrifted)
            if fluctuate:
                qsum = qblock.data.sum(dim=(-1,-2), keepdim=True)
                variations = torch.randn_like(qblock.data)
                qblock.data += variations
                qblock.data /= qsum

            signal = chunksum(qblock)

            # Nqblock = signal.nbatches
            iblock = convo(signal, response)
            current = chunksum_i(iblock)
            readout_segmetns = chunksum_readout(current)
            concatenate_waveforms(readout_segmetns, Nt=19200)

    if device == 'cuda':
        torch.cuda.synchronize()
    tend = time.time()
    info(f'Total elapsed time {tend - tstart} seconds')

    try:
        if export_pickle:
            torch.cuda.memory._dump_snapshot(f"graph_effq.pickle")
    except Exception as e:
        log.error(f"Failed to capture memory snapshot {e}")
    torch.cuda.memory._record_memory_history(enabled=None)

    if device == 'cuda':
        info(f"Peak memory usage {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")

    return

def fdsim(config, finpath, foutpath):
    torch.set_num_threads(1)
    global response_path
    global lifetime
    global drtoa
    global tspace
    global save_waveform
    global fluctuate

    global input_path
    global output_path

    global response

    with open(config, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    response_path = config.get("response_path",  "response_v2a_distance_10p431cm_binsize_0p04434cm_tick0p05us.npy")
    tspace = config.get("tspace", 0.5) * units.us/ units.us # values are in units of us
    lifetime = config.get("lifetime", 2.0) * units.ms / units.us # values are from ms units of us
    save_waveform = config.get("save_waveform", False)
    fluctuate = config.get("fluctuate", False)
    if fluctuate:
        torch.manual_seed(10)

    # loading response
    if os.path.splitext(response_path)[1] == '.npz':
        fres = np.load(response_path)
        response = ndlarsim(fres['response'])
        # tspace = fres['time_tick']  * units.us / units.us # us
        drtoa = fres['drift_length'] * units.cm / units.cm # cm
    else:
        raise ValueError('Only npz response is supported currently.')
    info(f'Loaded response for NDLAr from {response_path} with shape {response.shape}.')
    # hand pick response to one row in pixel space
    half_pos = response.shape[0]//2
    # let us transform from 0.05us to 0.5us
    response = response[half_pos, :, :]
    response = torch.mean(response.view(-1, response.size(-1)//10, 10), dim=-1)
    # pad additional 80 zeros to left and right in the first dim
    response = nn.functional.pad(response, (0, 0, 80, 80))
    # add random numbers so that convolution output are not dropped
    response[:80] = torch.randn(80, response.size(-1)) * 0.01*torch.max(response)
    response[-80:] = torch.randn(80, response.size(-1)) * 0.01*torch.max(response)

    info(f'Response shape after transforming to 0.5us: {response.shape}')

    if finpath is None:
        input_path = "depos.npy"
    else:
        input_path = finpath

    if foutpath is None:
        output_path = "waveforms.npz"
    else:
        output_path = foutpath

    # with torch.no_grad():
    #     runit('cuda')
    with torch.no_grad():
        runit('cpu')
