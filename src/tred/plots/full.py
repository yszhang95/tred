#!/usr/bin/env python
'''
Plots to exercise a full chain.

This combines parts of drift, raster, sparse, convo
'''

from tred import units
from tred.drift import drift
from tred.raster.depos import binned as raster_depos
from .response import get_ndlarsim
from tred.sparse import chunkify
from tred.blocking import Block
from tred.partitioning import deinterlace
from tred.convo import interlaced
import torch

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
    return depos.T


def do_drift(depos, velocity = -1.6 * units.mm/units.us):
    '''
    This mocks up some drift parameters, calls and repacks into depo form.
    '''
    depos = depos.T
    t = depos[0]
    q = depos[1]
    xyz = depos[2:5].T

    # some made up physics constants
    DL = 7.2 * units.cm2/units.s
    DT = 12.0 * units.cm2/units.s
    diffusion = torch.tensor([DL, DT, DT])

    lifetime = 8*units.ms
    
    centers, times, sigmas, charges = drift(xyz, velocity, diffusion, lifetime, times=t, charge=q)

    # repack to depo form.  This is unnecessary and wasteful for the real
    # program but here it helps to keep the return value tidy.
    drifted = torch.vstack((times, charges, centers.T, sigmas.T)).T
    return drifted

def do_raster(drifted, velocity = -1.6 * units.mm/units.us, nimperpix=10):
    '''
    Mock up the rastering stage.

    In this we make a dimension transformation from x,y,z to y,z,t.  This is a
    choice that must somehow be handled in the real application.  Perhaps given
    to user control.  The "taxis" argument to post-raster functions can in
    principle handle other dimension permutation choices.
    '''

    drifted = drifted.T

    t = drifted[0]
    y = drifted[4]
    z = drifted[5]
    centers = torch.vstack((y,z,t)).T

    sigmaL,sigmaT = drifted[6:]
    # Convert sigma_x to sigma_t
    sigmaL /= abs(velocity) 
    sigmas = torch.vstack((sigmaT,sigmaT,sigmaL)).T

    charges = drifted[1]

    # Here we define the mapping from real coordinates to the (fine) grid
    # indices.  In the real application, this must be a user config and
    # consistent to both response and depo data.  It must be consistent with
    # later chosen super-grid spacing.  
    pitch = 4.4*units.mm
    pspace = pitch/nimperpix
    tspace = 50*units.us
    grid_spacing = (pspace, pspace, tspace)

    # This is a somewhat arbitrary choice for the truncation of the diffusion
    # Gaussian.
    nsigma = torch.tensor([3.0]*3)

    # note, volume dimensions are now (y,z,t)

    # fixme: make raster functions return Block
    rasters, offsets = raster_depos(grid_spacing, centers, sigmas, charges, nsigma=nsigma)

    block = Block(location = offsets, data=rasters)
    print(f'rastered to {block.nbatches} blocks of shape {block.shape}')
    return block

def plot_full_3d(out):

    # what is best practice to assure all tensors on are a desired device?
    device = 'cpu'

    velocity = -1.6 * units.mm/units.us

    # The number of impact positions per pixel aka the "interlace step size".
    nimperpix = 10

    # some initial depos as batched 7-tuples
    depos = make_depos(device)
    print(f'made {depos.shape[0]} depos on {depos.device}')

    # same batched 7-tuple form after drifting
    drifted = do_drift(depos, velocity=velocity)
    print(f'drifted {drifted.shape[0]} depos on {depos.device}')

    # The variable name refers Yousen's name "universal block" which a batched
    # 1+3D tensor of arbitrary 3D shape and is unaligned to any pixels and holds
    # rastered ionization.  The volume dimensions are (y,z,t).
    uniblock = do_raster(drifted, velocity=velocity, nimperpix=nimperpix)

    # determine the super-grid.  To be pixel aligned, it must have a multiple of
    # the nimperpix on the space dimensions.  This choice can be optimized given
    # nominal ionization patterns.  There is a competing balance to minimize the
    # number of super-grid points (maximize bin size) while also maximizing bin
    # density (minimize zeros).
    npixpersuper = 10
    # The choice for the super-grid spacing along the time/drift dimensions is
    # rather more free.  But, it is also subject to some kind of min/max
    # optimization to like above.  Nominally, we start with the same number of
    # grid points.
    ntickperslice = 100

    chunk_size = (npixpersuper * nimperpix, npixpersuper * nimperpix, ntickperslice)

    sig = chunkify(uniblock, chunk_size)
    print(f'chunked to {sig.nbatches} blocks of shape {sig.shape} on {sig.data.device}')

    res = get_ndlarsim()
    print(f'response of shape {res.shape} on {res.device}')

    lacing = torch.tensor([nimperpix, nimperpix, 1])
    taxis = -1                  # the time axis
    meas = interlaced(sig, res, lacing, taxis)
    print(f'{meas.shape=}')
    #- [ ] make dense pixel waveforms...
    #- [ ] apply readout




def plots(out):
    with torch.no_grad():
        plot_full_3d(out)

