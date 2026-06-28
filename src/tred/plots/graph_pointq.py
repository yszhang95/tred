#!/usr/bin/env python
from tred.graph import Drifter, Raster, ChunkSum, LacedConvo, Charge, Current, Sim
from tred.response import ndlarsim
from tred.blocking import Block, concat_blocks, iter_chunk_block
from tred import units
from .response import get_ndlarsim
from tred.util import debug, info, tenstr, warning, iter_tensor_chunks
from tred.io_nd import simple_geo_parser, tpc_drift_direction
from tred.io import write_npz
from tred import chunking
from tred.readout import nd_readout

import sys
import h5py
import numpy as np
import yaml
from collections import defaultdict
from types import SimpleNamespace
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
threshold = None
event_list = None
save_waveform = None
const_recomb = None

npoints = None

# point-charge generation
point_charge = None  # ke- per iteration
niter = None         # number of iterations to simulate and save
tpc_id = None        # which TPC to drop the point charge into

uncorr_noise = None
reset_noise = None
thres_noise = None
fluctuate = False
effq_out_nt = 1

pitch = 4.434*units.mm / units.cm # values are in units of cm
nimperpix=10
pspace = pitch/nimperpix
velocity = 1.59645 * units.mm/units.us / (units.cm/units.us) # values are in units of cm/us

adc_hold_delay = None
adc_down_time = None
csa_reset_time = None
one_tick = None

response = None

old_geo_config = True

convo_o_shape = None
benchmark_each_stage = True
batch_scheme = [100, 50]
record_op_w_max_mem = True

def update_peak_memory_label(op_name, peak_mem_mb, current_label):
    """Return updated (peak_mem_mb, label) if current stage exceeds peak."""
    if not record_op_w_max_mem or not torch.cuda.is_available():
        return peak_mem_mb, current_label
    current_peak = torch.cuda.memory.max_memory_allocated() / 1024**2
    if current_peak > peak_mem_mb:
        return current_peak, op_name
    return peak_mem_mb, current_label

def load_threshold(threshold):
    '''
    A map of io group from data to MC should be done.
    Hard-coded map is provided for 2x2 geometry.
    Assume thresholds are aligned from lower to high ends.
    '''
    if not isinstance(threshold, str):
        return [torch.tensor(threshold), ] * 1000 # FIXME: A large enough number
    thresholds = []
    # io_groups = [1, 2, 3, 4, 5, 6, 7, 8]
    # io_indices_tred = [1, 0, 3, 2, 5, 4, 7, 6]
    with h5py.File(threshold, 'r') as fthres:
        for ig in [2,1,4,3,6,5,8,7]:
            thresholds.append(torch.tensor(fthres[f'io_group{ig}/threshold']['Q'][:], dtype=torch.float32))
    return thresholds

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
    Build per-TPC geometry (no input file needed). Each entry exposes the same
    attributes used downstream as a TPCDataset: tpc_id, drift, anode, cathode,
    lower_left_corner, upper_corner.
    '''
    borders = simple_geo_parser(module_yaml, tile_yaml, old_geo_config)
    anodes, cathodes, drifts, lowers, uppers = tpc_drift_direction(borders)
    tpcs = []
    for i in range(len(anodes)):
        tpcs.append(SimpleNamespace(
            tpc_id=i, drift=int(drifts[i]),
            anode=anodes[i], cathode=cathodes[i],
            lower_left_corner=lowers[i], upper_corner=uppers[i]))
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
    export_pickle = False

    # eventually replace this hard-wire with configuration
    # twindow_max = 12_000 # 12_000 * 50ns = 600us
    twindow_max = 7_200 # 12_000 * 50ns = 600us
    DL = 6.6270 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    DT = 13.2427 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    diffusion = torch.tensor([DL, DT, DT])
    grid_spacing = (pspace, pspace, tspace)
    npixpersuper = 4  # 8+1-5
    ntickperslice = 32
    chunk_shape = (npixpersuper * nimperpix, npixpersuper * nimperpix, ntickperslice)

    efield = 0.5 # kV/cm
    rho = 1.38 # g/cm^3
    A3t = 0.8 # birks
    k3t = 0.0486 # (g/MeV cm^2) (kV/cm); birks
    Wi = 23.6E-6 # MeV/pair

    lacing = torch.tensor([nimperpix, nimperpix, 1])

    batch_size = 4096*8

    t0 = time.time()
    # create intermediate nodes
    # dummy drifter for running time tests
    drifter = Drifter(diffusion, lifetime, velocity, drtoa=drtoa)
    raster = Raster(velocity, grid_spacing)
    chunksum = ChunkSum(chunk_shape)

    cshape_effq_out = torch.tensor([nimperpix, nimperpix, effq_out_nt])

    chunksum_effq_out = ChunkSum(cshape_effq_out) # 1 pixel, 1 pixel, 60*0.05us*1.6cm/us=4.8mm

    chunksum_readout = ChunkSum((1,1,120))
    # de-laced signal (4,4,32) and de-laced response (9,9,*): c_shape = (12,12,*);
    # time padded to 4096 (>= c_shape, multiple of chunksum_i time chunk 32).
    convo = LacedConvo(lacing, o_shape=(4+9-1, 4+9-1, 4096))
    # FIXME: (4, 4, 32) is a common divider of chunk_shape and convo_o_shape
    chunksum_i = ChunkSum((4, 4, 32), method='chunksum_inplace_v2')

    chunksum_i = chunksum_i.to(device)
    chunksum_readout = chunksum_readout.to(device)
    chunksum_effq_out = chunksum_effq_out.to(device)

    t1 = time.time()

    # response = ndlarsim(response_path) # response is loaded in main function

    global response
    response = response.to(device=device)

    t2 = time.time()

    tpcs = make_nd('cpu')

    t3 = time.time()

    runtime = defaultdict(list)

    waveforms = {}
    all_hits = []   # rows: [iter, pix_y, pix_z, time_tick, x, y, z, charge_ke]
    all_effq = []   # rows: [iter, pix_y, pix_z, total_effq_ke]  (per-pixel true charge, summed over time)
    truth = []      # rows: [iter, x, y, z, charge_dep_ke, charge_quench_ke]

    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    # MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000
    if export_pickle:
        torch.cuda.memory._record_memory_history(
        # max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )


    thresholds = load_threshold(threshold)

    info('Batch scheme: ' + str(batch_scheme))

    for itpc, tpcdataset in enumerate(tpcs):
        if tpcdataset.tpc_id != tpc_id:
            continue
        info(f"Drift direction: {tpcdataset.drift} in tpcid {tpcdataset.tpc_id}.")
        info(f"TPC lower corner: {tpcdataset.lower_left_corner} in itpc {tpcdataset.tpc_id}.")
        info(f"TPC upper corner: {tpcdataset.upper_corner} in itpc {tpcdataset.tpc_id}.")
        info(f"TPC anode: {tpcdataset.anode} in itpc {tpcdataset.tpc_id}.")
        info(f"TPC cathode: {tpcdataset.cathode} in itpc {tpcdataset.tpc_id}.")
        drifter = Drifter(diffusion, lifetime, tpcdataset.drift*velocity, fluctuate=fluctuate,
                          target=tpcdataset.anode, drtoa=drtoa)
        drifter = drifter.to(device=device)

        raster = Raster(tpcdataset.drift*velocity, grid_spacing, npoints=npoints).to(device=device)
        # raster = raster.to(device=device)
        chunksum = chunksum.to(device=device)
        convo = convo.to(device=device)

        tpc_lower_left = tpcdataset.lower_left_corner.to(device).unsqueeze(0)
        waveforms[f'tpc_lower_left_tpc{tpcdataset.tpc_id}'] = tpc_lower_left.cpu().squeeze(0)
        waveforms[f'tpc_upper_tpc{tpcdataset.tpc_id}'] = tpcdataset.upper_corner.cpu()
        waveforms[f'drift_direction_tpc{tpcdataset.tpc_id}'] = tpcdataset.drift
        waveforms[f'tpc_anode_tpc{tpcdataset.tpc_id}'] = tpcdataset.anode
        waveforms[f'tpc_cathode_tpc{tpcdataset.tpc_id}'] = tpcdataset.cathode
        waveforms[f'pixel_pitch_tpc{tpcdataset.tpc_id}'] = pitch


        inds_range = (tpcdataset.upper_corner - tpcdataset.lower_left_corner) // pitch
        inds_range = inds_range.to(torch.int32).to(device)

        # TPC center, absolute coords (cm): x fixed at drift center, y/z transverse
        xc = float((tpcdataset.anode + tpcdataset.cathode) / 2)
        yc = float((tpcdataset.lower_left_corner[0] + tpcdataset.upper_corner[0]) / 2)
        zc = float((tpcdataset.lower_left_corner[1] + tpcdataset.upper_corner[1]) / 2)

        for ibatch in range(niter):

            stime = time.time()
            peak_mem = 0
            op_w_max_mem = 'NoOp'
            try:
                # point charge created at t=0 -> global time reference is zero
                global_tref = [0.0, 0.0]

                # enable when benchmark each stage
                if device == 'cuda' and benchmark_each_stage:
                    torch.cuda.synchronize()
                t00 = time.time()

                if device == 'cuda' and benchmark_each_stage:
                    torch.cuda.synchronize()
                t01 = time.time()

                # point charge at TPC center with transverse-only offset, uniform
                # over one pixel pitch: [-pitch/2, +pitch/2)
                dy = (torch.rand(1).item() - 0.5) * pitch
                dz = (torch.rand(1).item() - 0.5) * pitch
                xtrue, ytrue, ztrue = xc, yc + dy, zc + dz

                charge = torch.tensor([point_charge * 1E3], dtype=torch.float32, device=device)  # ke- -> e-
                local_time = torch.zeros(1, dtype=torch.float32, device=device)
                tail = torch.tensor([[xtrue, ytrue, ztrue]], dtype=torch.float32, device=device)
                head = tail.clone()  # zero-length step: head == tail
                tail[:,[1,2]] -= tpc_lower_left
                head[:,[1,2]] -= tpc_lower_left

                if device == 'cuda' and benchmark_each_stage:
                    torch.cuda.synchronize()
                t02 = time.time()

                # steps path: head is given so the point is rasterized with the
                # Gauss-Legendre quadrature rule (npoints).
                drifted = drifter(local_time, charge, tail, head)
                drifted = list(d for d in drifted)
                min_sigma = torch.tensor([[tspace*abs(velocity)/2,
                                           pitch/10/2, pitch/10/2]]).to(device)
                drifted[0] = torch.clamp(drifted[0], min=min_sigma)

                # truth: deposited charge and quenched charge surviving lifetime
                # absorption at the anode (drifted[2] is post-absorption, in e-)
                q_quench = float(drifted[2].sum().item()) / 1E3  # e- -> ke-
                truth.append([ibatch, xtrue, ytrue, ztrue, float(point_charge), q_quench])

                if device == 'cuda' and benchmark_each_stage:
                    torch.cuda.synchronize()
                t03 = time.time()

                nbchunk = batch_scheme[0]

                current_blocks = []
                effq_blocks = []
                Nqblock = 0

                peak_mem, op_w_max_mem = update_peak_memory_label('before_drifter', peak_mem, op_w_max_mem)

                dt04 = 0  # effq
                dt05 = 0  # chunksum q
                dt06 = 0  # convo
                dt07 = 0  # chunksum i
                for ichunk, idrifted in enumerate(
                        iter_tensor_chunks(drifted, chunk_size=nbchunk)):
                    if device == 'cuda' and benchmark_each_stage:
                        torch.cuda.synchronize()
                    t04 = time.time()
                    qblock = raster(*idrifted)
                    if device == 'cuda' and benchmark_each_stage:
                        torch.cuda.synchronize()
                    dt04 += time.time() - t04
                    if record_op_w_max_mem:
                        peak_mem, op_w_max_mem = update_peak_memory_label('rasterization', peak_mem, op_w_max_mem)
                    # Check whether there is not-a-value elements.
                    assert ~torch.any(torch.isnan(qblock.data))
                    if device == 'cuda' and benchmark_each_stage:
                        torch.cuda.synchronize()
                    t05 = time.time()
                    signal = chunksum(qblock)
                    if device == 'cuda' and benchmark_each_stage:
                        torch.cuda.synchronize()
                    dt05 += time.time() - t05
                    if record_op_w_max_mem:
                        peak_mem, op_w_max_mem = update_peak_memory_label('chunksum_raster', peak_mem, op_w_max_mem)

                    # effective charge per pixel: sum rasterized (post-quench)
                    # charge over each pixel footprint; index -> pixel units
                    effqb = chunksum_effq_out(qblock)
                    effqb.location[:, 0:2] //= nimperpix
                    effq_blocks.append(effqb)
                    qblock = None
                    effqb = None
                    Nqblock += signal.nbatches

                    # if device == 'cuda':
                    #     torch.cuda.synchronize()
                    # t05 = time.time()

                    currents = []
                    for iqblock in iter_chunk_block(signal, chunk_size=batch_scheme[1]):
                        if iqblock.nbatches == 0:
                            continue
                        if device == 'cuda' and benchmark_each_stage:
                            torch.cuda.synchronize()
                        t06 = time.time()
                        iblock = convo(iqblock, response)
                        if device == 'cuda' and benchmark_each_stage:
                            torch.cuda.synchronize()
                        dt06 += time.time() - t06
                        if record_op_w_max_mem:
                            peak_mem, op_w_max_mem = update_peak_memory_label('convo', peak_mem, op_w_max_mem)
                        if device == 'cuda' and benchmark_each_stage:
                            torch.cuda.synchronize()
                        t07 = time.time()
                        current = chunksum_i(iblock)
                        if device == 'cuda' and benchmark_each_stage:
                            torch.cuda.synchronize()
                        dt07 += time.time() - t07
                        if record_op_w_max_mem:
                            peak_mem, op_w_max_mem = update_peak_memory_label('chunksum_i', peak_mem, op_w_max_mem)
                        currents.append(current)

                    # no need to chunk again; just sum
                    if device == 'cuda' and benchmark_each_stage:
                        torch.cuda.synchronize()
                    t07 = time.time()
                    currents = concat_blocks(currents)
                    if currents is not None:
                        currents = chunking.accumulate(currents)
                        if device == 'cuda' and benchmark_each_stage:
                            torch.cuda.synchronize()
                        current_blocks.append(currents)
                    dt07 += time.time() - t07
                    if record_op_w_max_mem:
                        peak_mem, op_w_max_mem = update_peak_memory_label('chunksum_i', peak_mem, op_w_max_mem)


                    # if device == 'cuda':
                    #     torch.cuda.synchronize()
                    # t05 = time.time()

                # per-pixel effective (true) charge: total over each pixel footprint and time
                effq = concat_blocks(effq_blocks)
                if effq is not None:
                    q_chunk = effq.data.sum(dim=(1, 2, 3))            # electrons per chunk
                    pix = effq.location[:, 0:2]                       # pixel index (y, z)
                    upix, inv = torch.unique(pix, dim=0, return_inverse=True)
                    q_pix = torch.zeros(upix.shape[0], device=q_chunk.device, dtype=q_chunk.dtype)
                    q_pix.scatter_add_(0, inv, q_chunk)
                    pmask = ((upix <= inds_range) & (upix >= 0)).all(dim=1)
                    upix = upix[pmask].cpu().to(torch.float32)
                    q_pix = (q_pix[pmask] / 1E3).cpu()               # e- -> ke-
                    if upix.shape[0] > 0:
                        itercol = torch.full((upix.shape[0], 1), float(ibatch))
                        all_effq.append(torch.cat([itercol, upix, q_pix[:, None]], dim=1))

                # no need to chunk again; just sum
                if device == 'cuda' and benchmark_each_stage:
                    torch.cuda.synchronize()
                t07 = time.time()
                currents = concat_blocks(current_blocks)
                if currents is not None:
                    currents = chunking.accumulate(currents)
                if device == 'cuda' and benchmark_each_stage:
                    torch.cuda.synchronize()
                dt07 += time.time() - t07
                if record_op_w_max_mem:
                    peak_mem, op_w_max_mem = update_peak_memory_label('chunksum_i', peak_mem, op_w_max_mem)

                if device == 'cuda':
                    torch.cuda.synchronize()
                t07 = time.time()

                if currents is None:
                    info(f'itpc{itpc}, tpc label {tpcdataset.tpc_id}, batch label {ibatch}, '
                         f'N qblock {Nqblock}, '
                         f'elapsed {t07 - stime} sec on {device}. Skipped empty batch.')
                    continue

                if device == 'cuda' and benchmark_each_stage:
                    torch.cuda.synchronize()
                t08 = time.time()  # currents for readout
                currents = chunksum_readout(currents)
                currents = concatenate_waveforms(currents, twindow_max, event_t=global_tref[1]//tspace)
                currents.data = currents.data * tspace / 1E3 # to ke-
                current_mask = (currents.location[:,[0,1]] <= inds_range) & (currents.location[:,[0,1]] >= 0)
                current_mask = current_mask.all(dim=1)
                currents = Block(data=currents.data[current_mask], location=currents.location[current_mask])
                if device == 'cuda' and benchmark_each_stage:
                    torch.cuda.synchronize()
                t09 = time.time()
                if record_op_w_max_mem:
                    peak_mem, op_w_max_mem = update_peak_memory_label('chunksum_readout', peak_mem, op_w_max_mem)

                if torch.isnan(currents.data).any():
                    raise ValueError

                # if isinstance(threshold, str):
                #     raise NotImplementedError("To add support for loading a threshold file.")
                thres = thresholds[tpcdataset.tpc_id].to(device)
                if thres.ndim > 0:
                    thres[thres<2] = 1E16 # FIXME: Temporarily disable low threshold channels
                hits = nd_readout(currents, thres, adc_hold_delay, adc_down_time, csa_reset_time, one_tick=one_tick,
                                  offset_to_align=0, # FIXME: how to calculate properly?
                                  pixel_axes=(1,2), uncorr_noise=uncorr_noise, thres_noise=thres_noise, reset_noise=reset_noise)
                if device == 'cuda':
                    torch.cuda.synchronize()
                t10 = time.time()
                if record_op_w_max_mem:
                    peak_mem, op_w_max_mem = update_peak_memory_label('readout', peak_mem, op_w_max_mem)


                runtime['to_device'].append(t01-t00)
                runtime['recomb'].append(t02-t01)
                runtime['drift'].append(t03-t02)
                runtime['raster'].append(dt04)
                runtime['chunksum_charge'].append(dt05)
                runtime['convo'].append(dt06)
                runtime['chunksum_current'].append(dt07)
                runtime['chunksum_readout'].append(t09 - t08)
                runtime['readout'].append(t10 - t09)

                info(f'{runtime["to_device"][-1]} data to {device}')
                info(f'{runtime["recomb"][-1]} recomb')
                info(f'{runtime["drift"][-1]} drift')
                info(f'{runtime["raster"][-1]} raster')
                info(f'{runtime["chunksum_charge"][-1]} chunksum_charge')
                info(f'{runtime["convo"][-1]} convo')
                info(f'{runtime["chunksum_current"][-1]} chunksum_current')
                info(f'{runtime["chunksum_readout"][-1]} chunksum_readout')
                info(f'{runtime["readout"][-1]} readout')

                info(f'Operation with max memory usage: {op_w_max_mem}')


                if device == 'cuda':
                    cuda_mem = torch.cuda.max_memory_allocated() / 1024**2
                    info(f'Peak cuda usage: {cuda_mem} MB')

                info(f'itpc{itpc}, tpc label {tpcdataset.tpc_id}, batch label {ibatch}, '
                      f'N qblock {Nqblock}, '
                      f'elapsed {t07 - stime} sec on {device}.')

                # save hits: raw pixel/time indices and transformed detector coords (cm)
                hitl = hits[0].cpu()
                if hitl.shape[0] > 0:
                    hoff = torch.tensor([1/2, 1/2, adc_hold_delay-global_tref[1]//tspace]).to(torch.float32)
                    hitlf32 = transform_indices_to_coord_3d(hitl[:,:3], pitch, tspace, velocity,
                                                            tpc_lower_left.to(torch.float32), tpcdataset.anode, tpcdataset.drift,
                                                            paxes=(0,1), taxis=-1, offset=hoff)
                    hitlf32 = hitlf32[:, [2,0,1]]  # (x, y, z)
                    itercol = torch.full((hitl.shape[0], 1), float(ibatch))
                    pix = hitl[:, :3].to(torch.float32)  # [pix_y, pix_z, time_tick]
                    qcol = hits[1][:,None].cpu().to(torch.float32)
                    all_hits.append(torch.cat([itercol, pix, hitlf32, qcol], dim=1))

                # if save_waveform and currents is not None:
                #     waveforms[f'current_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = currents.data.cpu().numpy()
                #     waveforms[f'current_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = currents.location.cpu().numpy()

                # # FIXME: global time offset
                # qbl = effq_blocks.location.to('cpu')
                # qoff = cshape_effq_out / 2
                # qoff[[0,1]] = qoff[[0,1]] / nimperpix
                # qoff[2] -= global_tref[1]//tspace
                # qblf32 = transform_indices_to_coord_3d(qbl, pitch, tspace, velocity,
                #                                        tpc_lower_left.to(torch.float32), tpcdataset.anode, tpcdataset.drift,
                #                                        paxes=(0,1), taxis=-1, offset=qoff)
                # qblf32 = qblf32[:, [2,0,1]]
                # qbd_fg = effq_blocks.data / 1E3 # to ke-
                # qbd = qbd_fg.sum(dim=(1,2,3))
                # qbd = torch.cat([qblf32, qbd[:,None]], dim=1)

                # hitl = hits[0].cpu()
                # # FIXME: :,:3 is hard-coded
                # hoff = torch.tensor([1/2, 1/2, adc_hold_delay-global_tref[1]//tspace]).to(torch.float32)
                # hitlf32 = transform_indices_to_coord_3d(hitl[:,:3], pitch, tspace, velocity,
                #                                         tpc_lower_left.to(torch.float32), tpcdataset.anode, tpcdataset.drift,
                #                                         paxes=(0,1), taxis=-1, offset=hoff)
                # hitlf32 = hitlf32[:, [2,0,1]]
                # hitd = torch.cat([hitlf32, hits[1][:,None].cpu()], dim=1)

                # waveforms[f'hits_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = hitd.numpy()
                # waveforms[f'hits_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = hitl.numpy()
                # waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = qbd
                # waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = qbl
                # waveforms[f'effq_fine_grain_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = qbd_fg
                # waveforms[f'effq_fine_grain_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = qbl

                torch.cuda.reset_peak_memory_stats()
            except IndexError as e:
                raise e

    # Stop recording memory snapshot history.
    # waveforms["tile_yaml"] = tile_yaml
    # waveforms["module_yaml"] = module_yaml
    # waveforms["response_path"] = response_path
    # waveforms["lifetime"] = lifetime
    # waveforms["drtoa"] = drtoa
    # waveforms["threshold"] = threshold
    # waveforms["event_list"] = event_list
    # waveforms["save_waveform"] = save_waveform
    # waveforms["uncorr_noise"] = uncorr_noise
    # waveforms["thres_noise"] = thres_noise
    # waveforms["reset_noise"] = reset_noise
    # waveforms["fluctuate"] = fluctuate
    # waveforms["effq_out_nt"] = effq_out_nt
    # waveforms["input_path"] = input_path
    # waveforms["adc_hold_delay"] = adc_hold_delay
    # waveforms["adc_down_time"] = adc_down_time
    # waveforms["csa_reset_time "] = csa_reset_time
    # waveforms["one_tick"] = one_tick
    waveforms[f'time_spacing'] = tspace

    # concatenated hits ([iter, pix_y, pix_z, time_tick, x, y, z, charge_ke])
    # and per-iteration truth
    waveforms['hits'] = (torch.cat(all_hits, dim=0).numpy() if all_hits
                         else np.zeros((0, 8), dtype=np.float32))
    # per-pixel effective (true) charge, summed over time: [iter, pix_y, pix_z, total_effq_ke]
    waveforms['effq'] = (torch.cat(all_effq, dim=0).numpy() if all_effq
                         else np.zeros((0, 4), dtype=np.float32))
    waveforms['truth'] = np.array(truth, dtype=np.float32)  # [iter,x,y,z,charge_dep_ke,charge_quench_ke]
    waveforms['point_charge'] = np.array(point_charge)
    waveforms['niter'] = np.array(niter)
    waveforms['tpc_id'] = np.array(tpc_id)

    write_npz(output_path, **waveforms)

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
    info(f'{sum(runtime["chunksum_readout"])} chunksum_readout')
    info(f'{sum(runtime["readout"])} readout')

    info(f'Total elapsed time {time.time() - t0} seconds')

    try:
        if export_pickle:
            torch.cuda.memory._dump_snapshot(f"graph_effq.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
    torch.cuda.memory._record_memory_history(enabled=None)

    info(f"Peak memory usage {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")

def plots(out):
    with torch.no_grad():
        # torch.set_default_device('cuda')
        # runit('cpu')
        # info('FINISHED CPU')
        runit('cuda')
        info('FINISHED CUDA')

def fullsim(config, finpath, foutpath):

    global tile_yaml
    global module_yaml
    global response_path
    global lifetime
    global drtoa
    global tspace
    global threshold
    global event_list
    global save_waveform
    global uncorr_noise
    global thres_noise
    global reset_noise
    global fluctuate
    global effq_out_nt
    global adc_hold_delay
    global adc_down_time
    global csa_reset_time
    global one_tick

    global const_recomb

    global npoints

    global old_geo_config

    global input_path
    global output_path

    global pspace
    global nimperpix
    global pitch

    global response

    global convo_o_shape
    global benchmark_each_stage
    global batch_scheme

    global point_charge
    global niter
    global tpc_id

    with open(config, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    tile_yaml = config.get('tile_yaml',  "tests/playground/multi_tile_layout-2.4.16.yaml")
    module_yaml = config.get('module_yaml',  "tests/playground/2x2_mod2mod_variation.yaml")
    response_path = config.get("response_path",  "response_v2a_distance_10p431cm_binsize_0p04434cm_tick0p05us.npy")
    drtoa = config.get("drtoa", 10.431) * units.cm / units.cm # values are in units of cm to cm
    tspace = config.get("tspace", 0.05) * units.us/ units.us # values are in units of us
    lifetime = config.get("lifetime", 2.0) * units.ms / units.us # values are from ms units of us
    threshold = config.get("threshold", 5.) # thousand electrons # it can also be a path to threshold
    event_list = config.get("event_list", None) # None means select all
    save_waveform = config.get("save_waveform", False)
    uncorr_noise = config.get("uncorr_noise", None)
    thres_noise = config.get("thres_noise", None)
    reset_noise = config.get("reset_noise", None)
    fluctuate = config.get("fluctuate", False)
    const_recomb = config.get("const_recomb", False)
    effq_out_nt = config.get("effq_out_nt", 1)
    old_geo_config = config.get("old_geo_config", True)
    convo_o_shape = config.get("convo_o_shape", (4, 4, 2048))
    backmark_each_stage = config.get("benchmark_each_stage", True)
    batch_scheme = config.get("batch_scheme", [100, 50])
    npoints = config.get('npoints', (2, 2, 2))

    # point-charge generation
    point_charge = config.get("point_charge", 100)  # ke- per iteration
    niter = config.get("niter", 100)                    # iterations to simulate and save
    tpc_id = config.get("tpc_id", 0)                    # which TPC to use

    # loading response
    if os.path.splitext(response_path)[1] == '.npz':
        fres = np.load(response_path)
        tspace = fres['time_tick']  * units.us / units.us # us
        drtoa = fres['drift_length'] * units.cm / units.cm # cm
        bin_size = fres["bin_size"] * units.cm / units.cm # cm
        warning(f'drtoa, tspace, will be overridden to {drtoa} cm, {tspace} us.')
        pspace = bin_size
        # use npath from the file if present, else config, else derive from pitch/bin_size
        if 'npath' in fres.files:
            nimperpix = int(fres['npath'])
        elif config.get('npath', None) is not None:
            nimperpix = int(config['npath'])
        else:
            nimperpix = int(round(pitch / bin_size))
            warning(f"response has no 'npath'; derived nimperpix={nimperpix} from pitch/bin_size.")
        pitch = pspace * nimperpix
        response = ndlarsim(fres['response'], nd_response_shape=fres['response'].shape[:2], nd_nimp=nimperpix)
    else:
        raise ValueError("Response must be in .npz file")

    adc_hold_delay = config.get("adc_hold_delay", 1.5) * units.us / units.us / (tspace * units.us / units.us)
    adc_hold_delay = int(round(adc_hold_delay))
    adc_down_time = config.get("adc_down_time", 1.2) * units.us / units.us / (tspace * units.us / units.us)
    adc_down_time = int(round(adc_down_time))
    csa_reset_time = config.get("csa_reset_time", 0.1) * units.us / units.us / (tspace * units.us / units.us)
    csa_reset_time = int(round(csa_reset_time))
    one_tick = config.get("one_tick", 0.1) * units.us / units.us / (tspace * units.us / units.us)
    one_tick = int(round(one_tick))

    if finpath is None:
        input_path = "/home/yousen/Public/ndlar_shared/data/tred_2x2_2025010/filtered_MiniRun5_1E19_RHC.convert2h5.0000000.EDEPSIM.hdf5"
    else:
        input_path = finpath

    if foutpath is None:
        output_path = "waveforms.npz"
    else:
        output_path = foutpath

    with torch.no_grad():
        runit('cpu')
