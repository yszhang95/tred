#!/usr/bin/env python
'''
Pytorch nn.Modules that call other modules or tred functions.

These "nodes" are used for:

- controlling use of CPU/GPU device
- providing a grammar to aggregate computation graphs
- providing cleavage lines to insert rebatching
- applying configuration and persisting data across executions

Only if a node needs to define parameters or receive configuration must it
implement __init__().

Developer take note:

- All quantities must be in tred system of units.

- Do NOT set device nor requires_grad explicitly.

- DO set device of temporary tensors based on device of established tensors.

'''

from .blocking import Block
from .drift import drift
from .util import debug, tenstr
from .raster.depos import binned as raster_depos
from .raster.steps import compute_qeff

from .types import index_dtype
from .sparse import chunkify
from .chunking import accumulate
from .convo import interlaced

import torch
import torch.nn as nn

def raster_steps(*args,**kwds):
    # raise NotImplementedError("Yousen, make raster.steps.function, import it as tred.graph.raster_steps and delete this temporary function")
    # rasters, offsets = raster_steps(self.grid_spacing, tail, head, sigma, charge, nsigma=self.nsigma)
    # args[0] : grid_spaing
    # args[1] : tail, X0
    # args[2] : tail, X1
    # args[3] : sigma
    # args[4] : charge, Q
    # kwds['nsigma] : nsigma, scalar
    return compute_qeff(grid_spacing=args[0], X0=args[1], X1=args[2],
                        Sigma=args[3], Q=args[4],
                        n_sigma=(kwds['nsigma'], kwds['nsigma'], kwds['nsigma']),
                        origin=(0,0,0), method='gauss_legendre', npoints=(2,2,2))

def param(thing, dtype=torch.float32):
    if isinstance(thing, torch.Tensor):
        return thing.to(dtype=dtype)

    return nn.Parameter(torch.tensor(thing, dtype=dtype), requires_grad=False)

def constant(node, name, thing, dtype=torch.float32):
    if isinstance(thing, torch.Tensor):
        thing = thing.to(dtype=dtype)
    else:
        thing = torch.tensor(thing, dtype=dtype)
    node.register_buffer(name, thing)

class Drifter(nn.Module):
    '''
    Drift charge along one of N-dimensions.
    '''
    def __init__(self, diffusion, lifetime, velocity,
                 target=0, vaxis=0, fluctuate=False):
        '''
        velocity is signed scalar
        vaxis is the axis on which the velocity is defined. default=0 (x)
        '''
        super().__init__()
        if target is None:
            raise ValueError('an absolute drift target is required (units [length])')
        if diffusion is None:
            raise ValueError('a diffusion coefficient tensor is required (units [length^2]/[time])')
        if lifetime is None:
            raise ValueError('a lifetime value is required (units [time])')
        if velocity is None:
            raise ValueError('a velocity value is required (units [distance]/[time])')

        constant(self, 'target', target)
        constant(self, 'diffusion', diffusion)
        constant(self, 'lifetime', lifetime)
        constant(self, 'velocity', velocity)
        constant(self, 'vaxis', vaxis, index_dtype)
        constant(self, 'fluctuate', fluctuate, bool)

    def forward(self, time, charge, tail, head=None):
        '''
        Drift depos or steps.

        Returns tuple one larger than input args with sigma prepended.
        '''

        # note, can in principle, drift with pare-existing sigmas.  But here we
        # assume point-like.
        dtail, dtime, dsigma, dcharge = drift(tail, velocity=self.velocity,
                                             diffusion=self.diffusion,
                                             lifetime=self.lifetime,
                                             target=self.target, times=time,
                                             vaxis=self.vaxis, charge=charge,
                                             fluctuate=self.fluctuate)
        if head is None:
            return (dsigma, dtime, dcharge, dtail)
        dhead = head + (dtail - tail)
        return (dsigma, dtime, dcharge, dtail, dhead)


class Raster(nn.Module):
    '''
    Raster depos or steps.
    '''
    def __init__(self, velocity, grid_spacing, pdims=(1,2), tdim=-1, nsigma=3.0):
        '''
        - pdims :: the N-1 dimensions for transverse pitch indexing into input points.
                   They are axes in the original tensor.
        - tdim :: the 1 dimension for time/drift.
                  This is the target axis of the new tensor to be processed.
        '''
        super().__init__()

        if velocity is None:
            raise ValueError('a velocity value is required (units [distance]/[time])')
        if grid_spacing is None:
            raise ValueError('a real-valued grid spacing N-tensor is required (units [distance])')

        constant(self, 'velocity', velocity)
        constant(self, 'grid_spacing', grid_spacing)
        constant(self, 'nsigma', nsigma)
        self._pdims = pdims or ()
        self._tdim = tdim

    def _time_diff(self, tail, head=None):
        """
        Always return (head - tail)/v if head is not None.

        taild and head are in a shape of (npt, vdim) or (npt,).
        """
        if head is None:
            return None
        ndims = len(self._pdims) + 1
        tdim = (set(range(ndims)) - set(self._pdims)).pop()
        d = tail - head if ndims == 1 else tail[:,tdim] - head[:,tdim]
        return d/self.velocity

    def _transform(self, point, time):
        """
        point is in a shape of (npt, vdim) or (npt,).

        The output point is always in a shape of (npt, vdim). When the input
        is in the shape of (npt,), the vdim for output is 1.
        """
        if point is None:
            return point

        ndims = len(self._pdims) + 1
        if self._pdims:
            old_tdim = (set(range(ndims)) - set(self._pdims)).pop()
            axes = list(iter(self._pdims))
            axes.insert(self._tdim, old_tdim)
            point = point[:, axes]

        point[:,self._tdim] = time

        return point

    def forward(self, sigma, time, charge, tail, head=None):
        '''
        Raster the input depos, return block.

        Input drifted charge undergoes a transformation of the tensor dimensions
        via pdims and tdim.

        If head is None then tail is a depo point.  Otherwise the two make a step.
        '''
        dt = self._time_diff(tail, head)

        tail = self._transform(tail, time)

        debug(f'grid:{tenstr(self.grid_spacing)} tail:{tenstr(tail)} sigma:{tenstr(sigma)} charge:{tenstr(charge)}')
        if head is None:        # depos, not steps
            rasters, offsets = raster_depos(self.grid_spacing, tail, sigma, charge, nsigma=self.nsigma)
            return Block(location = offsets, data=rasters)

        head = self._transform(head, dt+time)
        rasters, offsets = raster_steps(self.grid_spacing, tail, head, sigma, charge, nsigma=self.nsigma)

        return Block(location = offsets, data=rasters)


class ChunkSum(nn.Module):
    '''
    Chunk a Block and sum values on common chunks.

    This typically should follow raster and convo.

    '''

    def __init__(self, chunk_shape=None):
        super().__init__()
        if chunk_shape is None:
            raise ValueError('a unitless, integer N-tensor chunk shape is required')
        constant(self, 'chunk_shape', chunk_shape, index_dtype)

    def forward(self, block: Block) -> Block:
        '''
        Return a new block chunked to given shape and with overlaps summed.
        '''
        # fixme: May wish to put each in its own module if dynamic rebatching helps.
        return accumulate(chunkify(block, self.chunk_shape))


class LacedConvo(nn.Module):
    '''
    Convolve an interlaced signal and a response.
    '''
    def __init__(self, lacing=None, taxis=-1):
        super().__init__()
        if lacing is None:
            raise ValueError('a unitless, integer N-tensor lacing is required')
        constant(self, 'lacing', lacing, index_dtype)
        self._taxis = taxis

    def forward(self, signal, response):
        '''
        Apply laced convolution.
        '''
        # fixme: allow for response to be pre-FFT'ed
        return interlaced(signal, response, self.lacing, self._taxis)

        
class Charge(nn.Module):
    def __init__(self, drifter, raster, chunksum):
        super().__init__()
        self.drifter = drifter
        self.raster = raster
        self.chunksum = chunksum

    def forward(self, time, charge, tail, head=None):
        drifted = self.drifter(time, charge, tail, head)
        qblock = self.raster(*drifted)
        return self.chunksum(qblock)


class Current(nn.Module):

    def __init__(self, convo, chunksum):
        super().__init__()
        self.convo = convo
        self.chunksum = chunksum

    def forward(self, signal, response):
        iblock = self.convo(signal, response)
        return self.chunksum(iblock)


class Sim(nn.Module):
    def __init__(self, charge, current):
        super().__init__()
        self.charge = charge
        self.current = current

    def forward(self, response, time, charge, tail, head=None):
        signal = self.charge(time, charge, tail, head)
        return self.current(signal, response)
