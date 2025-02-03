#!/usr/bin/env python
from tred.graph import Drifter, Raster, ChunkSum, LacedConvo, Charge, Current, Sim
from tred import units
from .response import get_ndlarsim
from tred.io import write_npz

import torch
import time

def dump(blk, name=""):
    if isinstance(blk, torch.Tensor):
        print(f'{blk.shape[0]} batchs of shape {blk.shape[1:]} on {blk.device} {name}')
        return

    print(f'{blk.nbatches} blocks of shape {blk.shape} on {blk.data.device} {name}')

def make_depos(device='cpu'):
    '''
    This mocks up some file of depo sets.
    '''
    # (t,q,x,y,z,sigmaL,sigmaT).
    depos = torch.tensor([
        [    0,     0,   0,    0,    0,    0], # 0,t
        [    1,     1,   1,    1,    1,    1], # 1,q
        [-10.0,  -5.0, 0.0,  5.0, 10.0, 20.0], # 2,x
        [-20.0, -10.0, 0.0, 10.0, 20.0, 40.0], # 3,y
        [-20.0, -10.0, 0.0, 10.0, 20.0, 40.0], # 4,z
        [    0,     0,   0,    0,    0,    0], # 5,sigmaL
        [    0,     0,   0,    0,    0,    0], # 6,sigmaT
    ], device=device)
    depos[0] *= units.ms
    depos[1] *= 1000*units.eplus
    depos[2:] *= units.mm
    depos = depos.to(device=device)
    return dict(time=depos[0], charge=depos[1], tail=depos[2:5].T)
    

def runit(device='cpu'):
    '''
    '''

    # eventually replace this hard-wire with configuration
    DL = 7.2 * units.cm2/units.s
    DT = 12.0 * units.cm2/units.s
    diffusion = torch.tensor([DL, DT, DT])
    lifetime = 8*units.ms
    velocity = -1.6 * units.mm/units.us
    pitch = 4.4*units.mm
    nimperpix=10
    pspace = pitch/nimperpix
    tspace = 50*units.us
    grid_spacing = (pspace, pspace, tspace)
    npixpersuper = 10
    ntickperslice = 100
    chunk_shape = (npixpersuper * nimperpix, npixpersuper * nimperpix, ntickperslice)
    lacing = torch.tensor([nimperpix, nimperpix, 1])

    t0 = time.time()
    # create intermediate nodes
    drifter = Drifter(diffusion, lifetime, velocity)
    raster = Raster(velocity, grid_spacing)
    chunksum = ChunkSum(chunk_shape)
    convo = LacedConvo(lacing)

    # create top-level node
    sim = Sim(Charge(drifter, raster, chunksum),
              Current(convo, chunksum)) 
    t1 = time.time()
    sim.to(device=device)
    t2 = time.time()
    depos = make_depos(device)
    dump(depos['tail'], "initial depo points")
    t3 = time.time()
    response = get_ndlarsim().to(device=device)
    dump(response, "detector response")
    t4 = time.time()
    current = sim(response, **depos)
    dump(current, "induced current")
    t5 = time.time()
    print(current.shape)
    write_npz("graph.npz", current = current, depos=depos)
    t6 = time.time()

    print(f'{t1-t0} construct')
    print(f'{t2-t1} to device')
    print(f'{t3-t2} make depos')
    print(f'{t4-t3} get response')
    print(f'{t5-t4} run sim')
    print(f'{t6-t5} save data')

def plots(out):
    with torch.no_grad():
        runit('cpu')
        print('FINISHED CPU')
        runit('cuda')    
        print('FINISHED CUDA')


        
