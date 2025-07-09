#!/usr/bin/env python
from tred.graph import Drifter, Raster, ChunkSum, LacedConvo, Charge, Current, Sim
from tred.response import ndlarsim
from tred.blocking import Block
from tred import units
from .response import get_ndlarsim
from tred.util import debug, info, tenstr, warning
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
    for i in range(Npix):
        off = offsets[i].item()
        if off > 0:
            wf_out[i, ..., :off] = 0

    # pm = torch.nonzero(loc_out[:,-1] < event_t)
    # if pm.numel() > 0:
    #     print('---------------------------found pm')
    #     pm = pm[0]
    #     tind = torch.arange(Nt, device=wf_out.device).view(1, Nt).expand(wf_out.shape).clone().detach()
    #     tpos = torch.abs(loc_out[:,-1] - event_t)
    #     for i in range(wf_out.ndim-1):
    #         tpos = tpos.unsqueeze(-1)
    #     tm = tind > tpos
    #     wf_out[pm] *= tm[pm]
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
    export_pickle = False

    # eventually replace this hard-wire with configuration
    twindow_max = 12_000 # 12_000 * 50ns = 600us
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

    t0 = time.time()
    # create intermediate nodes
    # dummy drifter for running time tests
    drifter = Drifter(diffusion, lifetime, velocity, drtoa=drtoa)
    raster = Raster(velocity, grid_spacing)
    chunksum = ChunkSum(chunk_shape)

    cshape_effq_out = torch.tensor([nimperpix, nimperpix, effq_out_nt])

    chunksum_effq_out = ChunkSum(cshape_effq_out) # 1 pixel, 1 pixel, 60*0.05us*1.6cm/us=4.8mm

    chunksum_readout = ChunkSum((1,1,120))
    # chunksum_readout = ChunkSum((1,1,12000))
    convo = LacedConvo(lacing, o_shape=(12, 12, 6912))
    # convo = LacedConvo(lacing, o_shape=(12, 12, 2048))
    chunksum_i = ChunkSum((4, 4, 128), method='chunksum_inplace_v2')

    chunksum_i = chunksum_i.to('cuda')
    chunksum_readout = chunksum_readout.to('cuda')
    chunksum_effq_out = chunksum_effq_out.to('cuda')

    t1 = time.time()

    # response = ndlarsim(response_path) # response is loaded in main function

    global response
    response = response.to(device=device)

    t2 = time.time()

    tpcs = segment_to_tpc(*make_nd('cpu'))

    t3 = time.time()

    runtime = defaultdict(list)

    waveforms = {}

    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    # MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000
    if export_pickle:
        torch.cuda.memory._record_memory_history(
        # max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )


    thresholds = load_threshold(threshold)

    for itpc, tpcdataset in enumerate(tpcs):
        info(f"Drift direction: {tpcdataset.drift} in tpcid {tpcdataset.tpc_id}.")
        info(f"TPC lower corner: {tpcdataset.lower_left_corner} in itpc {tpcdataset.tpc_id}.")
        info(f"TPC upper corner: {tpcdataset.upper_corner} in itpc {tpcdataset.tpc_id}.")
        info(f"TPC anode: {tpcdataset.anode} in itpc {tpcdataset.tpc_id}.")
        info(f"TPC cathode: {tpcdataset.cathode} in itpc {tpcdataset.tpc_id}.")
        sampler = SortedLabelBatchSampler(tpcdataset.labels[:,0], batch_size)
        loader = CustomNDLoader(tpcdataset, sampler=sampler,
                                batch_size=None, collate_fn=nd_collate_fn)
        drifter = Drifter(diffusion, lifetime, tpcdataset.drift*velocity, fluctuate=fluctuate,
                          target=tpcdataset.anode, drtoa=drtoa)
        drifter = drifter.to(device=device)

        raster = Raster(tpcdataset.drift*velocity, grid_spacing).to(device=device)
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

        for ibatch, (features, labels) in enumerate(loader):

            stime = time.time()
            try:
                if isinstance(event_list, list) and len(event_list)>0 and int(labels[0,0].numpy()) not in event_list:
                    continue

                global_tref = [features[0][0,-2].numpy(), torch.min(features[0][:,-1]).numpy()] # assume it is in us
                waveforms[f'global_tref_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = np.array(global_tref)
                waveforms[f'event_id_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = labels[0,0].numpy()
                # assume there is only one particle in the event
                waveforms[f'event_start_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = features[0][0,2:5].numpy()
                waveforms[f'event_end_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = features[0][-1,5:8].numpy()


                if device == 'cuda':
                    torch.cuda.synchronize()
                t00 = time.time()
                features = [f.to(device=device) for f in features]

                if device == 'cuda':
                    torch.cuda.synchronize()
                t01 = time.time()

                charge = birks(dE=features[0][:,0], dEdx=features[0][:,1],
                          efield=efield, rho=rho, A3t=A3t, k3t=k3t, Wi=Wi)
                if const_recomb:
                    charge = features[0][:,0] / Wi * const_recomb # MeV / MeV/pair

                if device == 'cuda':
                    torch.cuda.synchronize()
                t02 = time.time()

                local_time = features[0][:,-1]
                tail = features[0][:,2:5]
                head = features[0][:,5:8]
                tail[:,[1,2]] -= tpc_lower_left
                head[:,[1,2]] -= tpc_lower_left

                # dsigma, dtime, dcharge, dtail, dhead
                # drifted = drifter(local_time, charge, tail, head)
                dsigma, dtime, dcharge, dtail, dhead = drifter(local_time, charge, tail, head)

                if device == 'cuda':
                    torch.cuda.synchronize()
                t03 = time.time()

                nbchunk = 100

                current_blocks_d = []
                current_blocks_l = []
                Nqblock = 0
                effq_blocks_d = []
                effq_blocks_l = []
                for ichunk in range(0, tail.size(0), nbchunk):
                    # print(ichunk, ibatch, 'log')
                    charge_this = charge[ichunk:ichunk+nbchunk]
                    qblock = raster(dsigma[ichunk:ichunk+nbchunk], dtime[ichunk:ichunk+nbchunk],
                                    dcharge[ichunk:ichunk+nbchunk], dtail[ichunk:ichunk+nbchunk], dhead[ichunk:ichunk+nbchunk])

                    # invalid = torch.isnan(qblock.data).any(dim=(1,2,3))

                    p0 = tail[ichunk:ichunk+nbchunk]
                    p1 = head[ichunk:ichunk+nbchunk]
                    length2 = torch.sum((p0-p1)**2, dim=1)
                    invalid2 = length2 < 1E-9
                    qblock.data[invalid2] = 0

                    signal = chunksum(qblock)
                    effqb = chunksum_effq_out(qblock)
                    effq_blocks_d.append(effqb.data.cpu())
                    effq_blocks_l.append(effqb.location.cpu())
                    effq_blocks_l[-1][:,0:2] //= nimperpix
                    effq_blocks_l[-1][:,-1] += int(abs(drtoa/velocity)//tspace)
                    qblock = None
                    effqb = None
                    Nqblock += signal.nbatches

                    # if device == 'cuda':
                    #     torch.cuda.synchronize()
                    # t05 = time.time()

                    nchunks = signal.nbatches // 50 + 1
                    ibs = []
                    currents_l = []
                    currents_d = []
                    currents = None
                    for l, s in zip(signal.location.chunk(nchunks), signal.data.chunk(nchunks)):
                        # print(s.shape[0])
                        if s.shape[0] > 0:
                            iblock = convo(Block(location=l, data=s), response)

                            current = chunksum_i(iblock)
                            currents_l.append(current.location)
                            currents_d.append(current.data)
                    # currents = Block(data=torch.cat(currents_d, dim=0).to('cpu'), location=torch.cat(currents_l, dim=0).to('cpu'))
                    if len(currents_l) > 0:
                        currents_d = torch.cat(currents_d, dim=0)
                        currents_l = torch.cat(currents_l, dim=0)
                        currents = Block(data=currents_d, location=currents_l)
                        currents = chunking.accumulate(currents)

                    if currents is not None:
                        # currents.data = currents.data.to('cpu')
                        # currents.location = currents.location.to('cpu')
                        current_blocks_d.append(currents.data)
                        current_blocks_l.append(currents.location)

                if len(current_blocks_d) > 0:
                    current_blocks_d = torch.cat(current_blocks_d, dim=0)
                    current_blocks_l = torch.cat(current_blocks_l, dim=0)
                    currents = Block(data=current_blocks_d, location=current_blocks_l)
                    currents = chunking.accumulate(currents)
                else:
                    currents = None

                t04 = t03
                t05 = t04
                if device == 'cuda':
                    torch.cuda.synchronize()
                t06 = time.time()

                if device == 'cuda':
                    torch.cuda.synchronize()
                t07 = time.time()

                if currents is None:
                    info(f'itpc{itpc}, tpc label {tpcdataset.tpc_id}, batch label {ibatch}, '
                         f'N segments {len(features[0])}, '
                         f'N qblock {Nqblock}, '
                         f'elapsed {t07 - stime} sec on {device}. Skipped empty batch.')
                    continue

                currents = chunksum_readout(currents)
                currents = concatenate_waveforms(currents, twindow_max, event_t=global_tref[1]//tspace)
                currents.data = currents.data * tspace / 1E3 # to ke-
                current_mask = (currents.location[:,[0,1]] <= inds_range) & (currents.location[:,[0,1]] >= 0)
                current_mask = current_mask.all(dim=1)
                currents = Block(data=currents.data[current_mask], location=currents.location[current_mask])

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
                      f'N qblock {Nqblock}, '
                      f'elapsed {t07 - stime} sec on {device}.')

                if save_waveform and currents is not None:
                    waveforms[f'current_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = currents.data.cpu().numpy()
                    waveforms[f'current_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = currents.location.cpu().numpy()

                # FIXME: global time offset
                qbl = torch.cat(effq_blocks_l).to('cpu')
                qoff = cshape_effq_out / 2
                qoff[[0,1]] = qoff[[0,1]] / nimperpix
                qoff[2] -= global_tref[1]//tspace
                qblf32 = transform_indices_to_coord_3d(qbl, pitch, tspace, velocity,
                                                       tpc_lower_left.to(torch.float32), tpcdataset.anode, tpcdataset.drift,
                                                       paxes=(0,1), taxis=-1, offset=qoff)
                qblf32 = qblf32[:, [2,0,1]]
                qbd_fg = torch.cat(effq_blocks_d) / 1E3 # to ke-
                qbd = qbd_fg.sum(dim=(1,2,3))
                qbd = torch.cat([qblf32, qbd[:,None]], dim=1)

                hitl = hits[0].cpu()
                # FIXME: :,:3 is hard-coded
                hoff = torch.tensor([1/2, 1/2, adc_hold_delay-global_tref[1]//tspace]).to(torch.float32)
                hitlf32 = transform_indices_to_coord_3d(hitl[:,:3], pitch, tspace, velocity,
                                                        tpc_lower_left.to(torch.float32), tpcdataset.anode, tpcdataset.drift,
                                                        paxes=(0,1), taxis=-1, offset=hoff)
                hitlf32 = hitlf32[:, [2,0,1]]
                hitd = torch.cat([hitlf32, hits[1][:,None].cpu()], dim=1)

                waveforms[f'hits_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = hitd.numpy()
                waveforms[f'hits_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = hitl.numpy()
                waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = qbd
                waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = qbl
                waveforms[f'effq_fine_grain_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = qbd_fg
                waveforms[f'effq_fine_grain_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = qbl

                torch.cuda.reset_peak_memory_stats()
            except IndexError as e:
                raise e

    # Stop recording memory snapshot history.
    waveforms["tile_yaml"] = tile_yaml
    waveforms["module_yaml"] = module_yaml
    waveforms["response_path"] = response_path
    waveforms["lifetime"] = lifetime
    waveforms["drtoa"] = drtoa
    waveforms["threshold"] = threshold
    waveforms["event_list"] = event_list
    waveforms["save_waveform"] = save_waveform
    waveforms["uncorr_noise"] = uncorr_noise
    waveforms["thres_noise"] = thres_noise
    waveforms["reset_noise"] = reset_noise
    waveforms["fluctuate"] = fluctuate
    waveforms["effq_out_nt"] = effq_out_nt
    waveforms["input_path"] = input_path
    waveforms["adc_hold_delay"] = adc_hold_delay
    waveforms["adc_down_time"] = adc_down_time
    waveforms["csa_reset_time "] = csa_reset_time
    waveforms["one_tick"] = one_tick
    waveforms[f'time_spacing'] = tspace

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

    info(f'Total elapsed time {time.time() - t0} seconds')

    try:
        if export_pickle:
            torch.cuda.memory._dump_snapshot(f"graph_effq.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
    torch.cuda.memory._record_memory_history(enabled=None)

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

    global old_geo_config

    global input_path
    global output_path

    global response

    with open(config, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    tile_yaml = config.get('tile_yaml',  "tests/playground/multi_tile_layout-2.4.16.yaml")
    module_yaml = config.get('module_yaml',  "tests/playground/2x2_mod2mod_variation.yaml")
    response_path = config.get("response_path",  "response_v2a_distance_10p431cm_binsize_0p04434cm_tick0p05us.npy")
    drtoa = config.get("drtoa", 10.431) * units.cm / units.cm # values are in units of cm to cm
    tspace = config.get("tspace", 0.05) * units.us/ units.us # values are in units of us
    lifetime = config.get("lifetime", 2.0) * units.ms / units.us # values are from ms units of us
    threshold = config.get("threshold", 5_000) # electrons # it can also be a path to threshold
    event_list = config.get("event_list", None) # None means select all
    save_waveform = config.get("save_waveform", False)
    uncorr_noise = config.get("uncorr_noise", None)
    thres_noise = config.get("thres_noise", None)
    reset_noise = config.get("reset_noise", None)
    fluctuate = config.get("fluctuate", False)
    const_recomb = config.get("const_recomb", False)
    effq_out_nt = config.get("effq_out_nt", 1)
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
        runit('cuda')
