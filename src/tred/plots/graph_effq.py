#!/usr/bin/env python
from tred.graph import Drifter, Raster, ChunkSum, LacedConvo, Charge, Current, Sim
from tred.response import ndlarsim
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
from tred import chunking
from tred.readout import nd_readout

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
    path = '/home/yousen/Public/ndlar_shared/data/segments_pid13.hdf5'
    # path = '/home/yousen/Public/ndlar_shared/data/segments_first_event.hdf5'
    d0 = StepLoader(h5py.File(path), transform=steps_from_ndh5)
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
    DL = 4.4 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    DT = 8.8 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    diffusion = torch.tensor([DL, DT, DT])
    lifetime = 2.2*units.ms / units.us # values are in units of us
    velocity = 1.6 * units.mm/units.us / (units.cm/units.us) # values are in units of cm/us
    pitch = 4.434*units.mm / units.cm # values are in units of cm
    nimperpix=10
    pspace = pitch/nimperpix
    tspace = 50*units.ns / units.us # values are in units of us
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

    threshold = 5_000
    adc_hold_delay = 30 # 30 * 50ns = 1.5us
    adc_down_time = 22 # 22 * 50ns = 1.1us
    csa_reset_time = 2 # 2 * 50ns = 100ns

    drtoa = 10.431 * units.cm / units.cm # values are in units of cm
    # drtoa = 50.4 * units.cm / units.cm # values are in units of cm

    lacing = torch.tensor([nimperpix, nimperpix, 1])

    batch_size = 4096

    t0 = time.time()
    # create intermediate nodes
    # dummy drifter for running time tests
    drifter = Drifter(diffusion, lifetime, velocity, drtoa=drtoa)
    raster = Raster(velocity, grid_spacing)
    chunksum = ChunkSum(chunk_shape)

    cshape_effq_out = torch.tensor([10, 10, 60])

    chunksum_effq_out = ChunkSum(cshape_effq_out) # 1 pixel, 1 pixel, 60*0.05us*1.6cm/us=4.8mm

    chunksum_readout = ChunkSum((1,1,12000))
    # convo = LacedConvo(lacing, o_shape=(12, 12, 6912))
    convo = LacedConvo(lacing, o_shape=(12, 12, 2048))
    chunksum_i = ChunkSum((4, 4, 128), method='chunksum_inplace_v2')

    chunksum_i = chunksum_i.to('cuda')
    # chunksum_readout = chunksum_readout.to('cuda')
    chunksum_effq_out = chunksum_effq_out.to('cuda')

    t1 = time.time()

    # response = get_ndlarsim()
    response = ndlarsim("response_v2a_distance_10p431cm_binsize_0p04434cm_tick0p05us.npy")

    response = response.to(device=device)

    t2 = time.time()

    tpcs = segment_to_tpc(*make_nd('cpu'))

    t3 = time.time()

    runtime = defaultdict(list)

    waveforms = {}

    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    # MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000
    torch.cuda.memory._record_memory_history(
        # max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


    for itpc, tpcdataset in enumerate(tpcs):
        sampler = SortedLabelBatchSampler(tpcdataset.labels[:,0], batch_size)
        loader = CustomNDLoader(tpcdataset, sampler=sampler,
                                batch_size=None, collate_fn=nd_collate_fn)
        drifter = Drifter(diffusion, lifetime, tpcdataset.drift*velocity,
                          target=tpcdataset.anode, drtoa=drtoa)
        drifter = drifter.to(device=device)

        raster = raster.to(device=device)
        chunksum = chunksum.to(device=device)
        convo = convo.to(device=device)

        tpc_lower_left = tpcdataset.lower_left_corner.to(device).unsqueeze(0)
        waveforms[f'tpc_lower_left_tpc{tpcdataset.tpc_id}'] = tpc_lower_left.cpu().squeeze(0)
        waveforms[f'tpc_upper_tpc{tpcdataset.tpc_id}'] = tpcdataset.upper_corner.cpu()
        waveforms[f'drift_direction_tpc{tpcdataset.tpc_id}'] = tpcdataset.drift
        waveforms[f'tpc_anode_tpc{tpcdataset.tpc_id}'] = tpcdataset.anode
        waveforms[f'tpc_cathode_tpc{tpcdataset.tpc_id}'] = tpcdataset.cathode
        waveforms[f'pixel_pitch_tpc{tpcdataset.tpc_id}'] = pitch
        waveforms[f'time_tick_tpc{tpcdataset.tpc_id}'] = tspace


        inds_range = (tpcdataset.upper_corner - tpcdataset.lower_left_corner) // pitch
        inds_range = inds_range.to(torch.int32)
        # print('ind_range', inds_range)

        # if itpc < 25:
        #     continue

        for ibatch, (features, labels) in enumerate(loader):
            stime = time.time()
            try:

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

                    # print('tail', tail[ichunk:ichunk+nbchunk][invalid])
                    # print('head', head[ichunk:ichunk+nbchunk][invalid])
                    # print('charge', charge_this[invalid])
                    # print('length', torch.sqrt(torch.sum((p1-p0)**2, dim=1))[invalid])
                    # d1 = dsigma[ichunk:ichunk+nbchunk]
                    # d2 = dtime[ichunk:ichunk+nbchunk]
                    # d3 = dcharge[ichunk:ichunk+nbchunk]
                    # d4 = dtail[ichunk:ichunk+nbchunk]
                    # d5 = dhead[ichunk:ichunk+nbchunk]
                    # print('qblock', torch.isnan(qblock.data).any())
                    # print('qblock data', qblock.location[torch.isnan(qblock.data).any(dim=(1,2,3))])
                    # print('qblock data', qblock.data[torch.isnan(qblock.data).any(dim=(1,2,3))])
                    # print('d1', d1[torch.isnan(qblock.data).any(dim=(1,2,3))])
                    # print('d2', d2[torch.isnan(qblock.data).any(dim=(1,2,3))])
                    # print('d3', d3[torch.isnan(qblock.data).any(dim=(1,2,3))])
                    # print('d4', d4[torch.isnan(qblock.data).any(dim=(1,2,3))])
                    # print('d5', d5[torch.isnan(qblock.data).any(dim=(1,2,3))])
                    # if device == 'cuda':
                    #     torch.cuda.synchronize()
                    # t04 = time.time()

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
                        currents.data = currents.data.to('cpu')
                        currents.location = currents.location.to('cpu')
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

                currents = chunksum_readout(currents)
                currents.data = currents.data * tspace
                uqt = torch.unique(currents.location[:,-1])
                uqpxl = torch.unique(currents.location[:,0:2], dim=0)
                # print('currents', torch.isnan(currents.data).any())
                if torch.isnan(currents.data).any():
                    raise ValueError
                # print('currents data', currents.location[torch.isnan(currents.data).any(dim=(1,2,3))])
                # info(f'----------------- unique pixels {currents.nbatches}')
                # info(f'----------------- unique time offsets {uqt.size(0)}')
                # info(f'----------------- unique pixel offsets {uqpxl.size(0)}')
                # info(f'----------------- max Q {torch.max(torch.sum(currents.data, dim=-1))}')

                hits = nd_readout(currents, threshold, adc_hold_delay, adc_down_time, csa_reset_time, pixel_axes=(1,2))

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

                # if currents is not None:
                #     waveforms[f'current_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = currents.data
                #     waveforms[f'current_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = currents.location
                # waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = signal



                # FIXME: global time offset
                qbl = torch.cat(effq_blocks_l).to('cpu')
                qoff = cshape_effq_out / 2
                qoff[[0,1]] = qoff[[0,1]] / nimperpix
                qblf32 = transform_indices_to_coord_3d(qbl, pitch, tspace, velocity,
                                                       tpc_lower_left.to(torch.float32), tpcdataset.anode, tpcdataset.drift,
                                                       paxes=(0,1), taxis=-1, offset=qoff)
                qblf32 = qblf32[:, [2,0,1]]
                qbd = torch.cat(effq_blocks_d).sum(dim=(1,2,3))
                qbd = torch.cat([qblf32, qbd[:,None]], dim=1)


                hitl = hits[0].cpu()
                mask = (hitl[:,[0,1]] <= inds_range) & (hitl[:,[0,1]] >= 0)
                mask = mask.all(dim=1)
                hitl = hitl[mask]
                # FIXME: :,:3 is hard-coded
                hoff = torch.tensor([1/2, 1/2, adc_hold_delay-global_tref[1]//tspace]).to(torch.float32)
                hitlf32 = transform_indices_to_coord_3d(hitl[:,:3], pitch, tspace, velocity,
                                                        tpc_lower_left.to(torch.float32), tpcdataset.anode, tpcdataset.drift,
                                                        paxes=(0,1), taxis=-1, offset=hoff)
                hitlf32 = hitlf32[:, [2,0,1]]
                hitd = torch.cat([hitlf32, hits[1][:,None].cpu()[mask]], dim=1)

                waveforms[f'hits_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = hitd.numpy()
                waveforms[f'hits_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = hitl.numpy()
                waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = qbd
                waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = qbl

                torch.cuda.reset_peak_memory_stats()
            except IndexError as e:
                raise e
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


    # Stop recording memory snapshot history.

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

    print('Porting memory usage')
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
