#!/usr/bin/env python
from tred.graph import Drifter, Raster, ChunkSum, LacedConvo, Charge, Current, Sim
from tred.response import ndlarsim
from tred.blocking import Block, concat_blocks, iter_chunk_block
from tred import units
from .response import get_ndlarsim
from tred.util import debug, info, tenstr, warning, iter_tensor_chunks
from tred.loaders import StepLoader, steps_from_ndh5
from tred.io_nd import (
    nd_collate_fn, create_tpc_datasets_from_steps,
    LazyLabelBatchSampler, EagerLabelBatchSampler, SortedLabelBatchSampler, CustomNDLoader, simple_geo_parser
)
from tred.recombination import birks, box
from tred.io import write_npz
from tred import chunking
from tred.readout import nd_readout

import torch.nn as nn
import torch.optim as optim

import sys
import h5py
import numpy as np
import yaml
from collections import defaultdict
import torch
import time
import os

import torch
import time

module_yaml = None
tile_yaml = None
response_path = None
lifetime = None
input_path = None
output_path = None
drtoa = None
tspace = None
event_list = None

pitch = 4.434*units.mm / units.cm # values are in units of cm
nimperpix=10
pspace = pitch/nimperpix
velocity = 1.59645 * units.mm/units.us / (units.cm/units.us) # values are in units of cm/us

response = None

old_geo_config = True

def params_eligible_this_step(optimizer, model):
    eligible = []
    for group in optimizer.param_groups:
        lr = group.get('lr', 0.0)
        for p in group['params']:
            if p.requires_grad and p.grad is not None and lr != 0.0:
                eligible.append(p)
    # Map back to names for readability
    names = {id(p): n for n, p in model.named_parameters()}
    return [names.get(id(p), f"<unnamed:{id(p)}>") for p in eligible]


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
    wf_out = wf_out.index_put((flat_batch, flat_time), flat_values, accumulate=True)


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
    # wf_out.masked_fill_(mask, 0)
    y = wf_out * (~mask).to(wf_out.dtype)

    return Block(data=y, location=loc_out)


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

def runit(device='cpu'):
    '''
    '''

    # lifetime_inc = 1.02
    # lifetime_dec = 0.98

    lifetime_inc = 1.2
    lifetime_dec = 0.8


    # eventually replace this hard-wire with configuration
    twindow_max = 12_000 # 12_000 * 50ns = 600us
    # twindow_max = 9_600 #
    DL = 4.0 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    DT = 8.8 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    diffusion = torch.tensor([DL, DT, DT])
    grid_spacing = (pspace, pspace, tspace)
    # npixpersuper = 16+1-9
    npixpersuper = 12+1-9
    # ntickperslice = 6912+1-6400
    ntickperslice = 384 # 128*3
    chunk_shape = (npixpersuper * nimperpix, npixpersuper * nimperpix, ntickperslice)

    efield = 0.5 # kV/cm
    rho = 1.38 # g/cm^3
    A3t = 0.8 # birks
    k3t = 0.0486 # (g/MeV cm^2) (kV/cm); birks
    Wi = 23.6E-6 # MeV/pair

    lacing = torch.tensor([nimperpix, nimperpix, 1])

    batch_size = 4096

    # create intermediate nodes
    # dummy drifter for running time tests
    drifter = Drifter(diffusion, lifetime, velocity, drtoa=drtoa)
    raster = Raster(velocity, grid_spacing)
    chunksum = ChunkSum(chunk_shape, method='chunksum_differential')

    chunksum_readout = ChunkSum((1,1,120), method='chunksum_differential')

    convo = LacedConvo(lacing, o_shape=(12, 12, 6912))
    chunksum_i = ChunkSum((4, 4, 128), method='chunksum_differential')

    chunksum_i = chunksum_i.to(device)
    chunksum_readout = chunksum_readout.to(device)

    # response = ndlarsim(response_path) # response is loaded in main function

    global response
    response = response.to(device=device)
    global selected_tpcid
    selected_tpcid = 2

    tpcs = segment_to_tpc(*make_nd('cpu'))

    total_losses = []
    lifetime_values = []
    edepsim = None
    nepoches = 1000

    lr = 0.8


    for itpc, tpcdataset in enumerate(tpcs):
        if selected_tpcid != tpcdataset.tpc_id:
            continue
        info("Selected TPC ID: {}".format(tpcdataset.tpc_id))

        # sampler = SortedLabelBatchSampler(tpcdataset.labels[:,0], batch_size)
        sampler = SortedLabelBatchSampler(tpcdataset.labels[:,0], 4096*16)
        loader = CustomNDLoader(tpcdataset, sampler=sampler,
                                batch_size=None, collate_fn=nd_collate_fn)
        drifter = Drifter(diffusion, lifetime, tpcdataset.drift*velocity,
                          target=tpcdataset.anode, drtoa=drtoa)
        drifter = drifter.to(device=device)

        raster = Raster(tpcdataset.drift*velocity, grid_spacing).to(device=device)
        chunksum = chunksum.to(device=device)
        convo = convo.to(device=device)

        chg = Charge(drifter, raster, chunksum)

        # sim = Sim(raster, )
        curr = Current(convo, chunksum_i)

        sim = Sim(chg, curr)
        sim = sim.to(device)

        tpc_lower_left = tpcdataset.lower_left_corner.to(device).unsqueeze(0)

        inds_range = (tpcdataset.upper_corner - tpcdataset.lower_left_corner) // pitch
        inds_range = inds_range.to(torch.int32).to(device)

        for ibatch, (features, labels) in enumerate(loader):
            if isinstance(event_list, list) and len(event_list)>0 and int(labels[0,0].numpy()) not in event_list:
                continue
            info(f'Processing event {int(labels[0,0].numpy())} batch {ibatch},'
                 f' {len(features[0])} segments ...')

            global_tref = [features[0][0,-2].numpy(), torch.min(features[0][:,-1]).numpy()] # assume it is in us
            features = [f.to(device=device) for f in features]

            charge = birks(dE=features[0][:,0], dEdx=features[0][:,1],
                      efield=efield, rho=rho, A3t=A3t, k3t=k3t, Wi=Wi)

            local_time = features[0][:,-1]
            tail = features[0][:,2:5]
            head = features[0][:,5:8]
            tail[:,[1,2]] -= tpc_lower_left
            head[:,[1,2]] -= tpc_lower_left

            edepsim = features[0].cpu().numpy()

            nbchunk = 20

            current_blocks = []
            with torch.no_grad():
                for ichunk, idata in enumerate(
                        iter_tensor_chunks((local_time, charge, tail, head),
                             chunk_size=nbchunk)):
                    # drifted = drifter(idata[0], idata[1], idata[2], idata[3])
                    # qblock = raster(*drifted)

                    induced_currents = sim(response, *idata)
                    currents = chunksum_readout(induced_currents)
                    currents = concatenate_waveforms(currents, twindow_max, event_t=global_tref[1]//tspace)
                    current_blocks.append(
                        Block(data=currents.data.detach().cpu(), location=currents.location.cpu())
                    )
                    currents = None
                    info(f'Peak GPU Memory Allocated: {torch.cuda.max_memory_allocated()/1024**3} GB in chunk {ichunk}')

                currents = concat_blocks(current_blocks)
                currents = chunking.accumulate(currents)
                currents_data_true = currents.data.detach().cpu()
                currents_location_true = currents.location.cpu()
                currents = None

            info(f'Peak GPU Memory Allocated before fit: {torch.cuda.max_memory_allocated()/1024**3} GB')

            currents_data_inc_0 = None
            currents_location_inc_0 = None
            currents_data_dec_0 = None
            currents_location_dec_0 = None

            # Training loop for lifetime * lifetime_inc
            sim.charge.drifter.lifetime = nn.Parameter(
                torch.tensor(lifetime*lifetime_inc, device=device), requires_grad=True)
            optimizer = optim.Adam([sim.charge.drifter.lifetime], lr=lr)
            for iepoch in range(nepoches):
                total_loss = 0
                cbs = []
                if iepoch != 0:
                    optimizer.zero_grad()

                for ichunk, idata in enumerate(
                    iter_tensor_chunks((local_time, charge, tail, head),
                         chunk_size=nbchunk)):
                    # info(f'Epoch {iepoch}, processing chunk {ichunk} ...')
                    induced_currents = sim(response, *idata)
                    currents = chunksum_readout(induced_currents)

                    currents = concatenate_waveforms(currents, twindow_max, event_t=global_tref[1]//tspace)
                    cbs.append(currents)

                currents = concat_blocks(cbs)
                loss = nn.MSELoss()(currents.data, currents_data_true.to(device=device))
                info(f'  Chunk {ichunk} loss: {loss.item()}')
                if iepoch != 0:
                    loss.backward()
                    # Call this after loss.backward() but before optimizer.step()
                    # print(params_eligible_this_step(optimizer, sim))
                    optimizer.step()
                else:
                    currents_data_inc_0 = currents.data.detach().cpu()
                    currents_location_inc_0 = currents.location.detach().cpu()
                total_loss += loss.item()
                info(f'Epoch {iepoch} total loss: {total_loss}')
                info(f'  Lifetime: {sim.charge.drifter.lifetime.item()*units.us/units.ms} ms')
                total_losses.append(total_loss)
                lifetime_values.append(sim.charge.drifter.lifetime.item()*units.us/units.ms)

            # Training loop for lifetime * lifetime_dec
            sim.charge.drifter.lifetime = nn.Parameter(
                torch.tensor(lifetime*lifetime_dec, device=device), requires_grad=True)
            optimizer = optim.Adam([sim.charge.drifter.lifetime], lr=lr)

            for iepoch in range(nepoches):
                total_loss = 0
                cbs = []
                if iepoch != 0:
                    optimizer.zero_grad()

                for ichunk, idata in enumerate(
                    iter_tensor_chunks((local_time, charge, tail, head),
                         chunk_size=nbchunk)):
                    # info(f'Epoch {iepoch}, processing chunk {ichunk} ...')
                    induced_currents = sim(response, *idata)
                    currents = chunksum_readout(induced_currents)

                    currents = concatenate_waveforms(currents, twindow_max, event_t=global_tref[1]//tspace)
                    cbs.append(currents)

                currents = concat_blocks(cbs)
                loss = nn.MSELoss()(currents.data, currents_data_true.to(device=device))
                info(f'  Chunk {ichunk} loss: {loss.item()}')
                if iepoch != 0:
                    loss.backward()
                    # Call this after loss.backward() but before optimizer.step()
                    # print(params_eligible_this_step(optimizer, sim))
                    optimizer.step()
                else:
                    currents_data_dec_0 = currents.data.detach().cpu()
                    currents_location_dec_0 = currents.location.detach().cpu()
                total_loss += loss.item()
                info(f'Epoch {iepoch} total loss: {total_loss}')
                info(f'  Lifetime: {sim.charge.drifter.lifetime.item()*units.us/units.ms} ms')
                total_losses.append(total_loss)
                lifetime_values.append(sim.charge.drifter.lifetime.item()*units.us/units.ms)

            info(f'Event {int(labels[0,0].numpy())} batch {ibatch} done.')

    return (total_losses, lifetime_values, edepsim,
            lifetime_inc, lifetime_dec,
            currents_data_true, currents_location_true,
            currents_data_inc_0, currents_location_inc_0,
            currents_data_dec_0, currents_location_dec_0)

    # Stop recording memory snapshot history.

def train(config, finpath):

    global tile_yaml
    global module_yaml
    global response_path
    global lifetime
    global drtoa
    global tspace
    global event_list

    global old_geo_config

    global input_path

    global response

    with open(config, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    tile_yaml = config.get('tile_yaml',  "tests/playground/multi_tile_layout-2.4.16.yaml")
    module_yaml = config.get('module_yaml',  "tests/playground/2x2_mod2mod_variation.yaml")
    response_path = config.get("response_path",  "response_v2a_distance_10p431cm_binsize_0p04434cm_tick0p05us.npy")
    drtoa = config.get("drtoa", 10.431) * units.cm / units.cm # values are in units of cm to cm
    tspace = config.get("tspace", 0.05) * units.us/ units.us # values are in units of us
    lifetime = config.get("lifetime", 2.0) * units.ms / units.us # values are from ms units of us
    event_list = config.get("event_list", None) # None means select all
    old_geo_config = config.get("old_geo_config", True)

    # loading response
    if os.path.splitext(response_path)[1] == '.npz':
        fres = np.load(response_path)
        response = ndlarsim(fres['response'])
        tspace = fres['time_tick']  * units.us / units.us # us
        drtoa = fres['drift_length'] * units.cm / units.cm # cm
        bin_size = fres["bin_size"] * units.cm / units.cm # cm
        warning(f'drtoa, tspace, will be overridden to {drtoa} cm, {tspace} us.')
        if abs(bin_size - pspace) > 1E-4:
            warning(f'Please manually check pspace. pspace in response file is {fres["bin_size"]} cm. pspace in config.')
    else:
        response = ndlarsim(response_path)

    if finpath is None:
        input_path = "/home/yousen/Public/ndlar_shared/data/tred_2x2_2025010/filtered_MiniRun5_1E19_RHC.convert2h5.0000000.EDEPSIM.hdf5"
    else:
        input_path = finpath

    # runit('cuda')
    total_losses, lifetime_values, edepsim, lifetime_inc, lifetime_dec, *currents = runit('cuda')

    np.savez("lifetime_fit_results.npz",
             total_losses=np.array(total_losses),
             lifetime_values=np.array(lifetime_values),
             edepsim=edepsim,
             lifetime_inc=lifetime_inc,
                lifetime_dec=lifetime_dec,
                currents_data_true=currents[0],
                currents_location_true=currents[1],
                currents_data_inc_0=currents[2],
                currents_location_inc_0=currents[3],
                currents_data_dec_0=currents[4],
                currents_location_dec_0=currents[5]
    )
