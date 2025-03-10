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

from .types import index_dtype
from .sparse import chunkify
from .chunking import accumulate
from .convo import interlaced

import torch
import torch.nn as nn

def raster_steps(*args,**kwds):
    raise NotImplementedError("Yousen, make raster.steps.function, import it as tred.graph.raster_steps and delete this temporary function")



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
                 target=0, vaxis=0, fluctuate=False,
                 tshift=None, drtoa=None):
        '''
        diffusion :: diffusion coefficients, real scalar (number or 0D tensor), or 1D tensor, or plain list/tuple.
                     The 1D tensor or plain list/tuple must be a sequence of numbers in a size of (vdim,)
        velocity is signed scalar
        vaxis is the axis on which the velocity is defined. default=0 (x)

        All parameters will be set to a constant tensor in the class instance.
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
        if tshift is not None and drtoa is not None:
            raise ValueError('tshift and drtoa are mutually exclusive arguments.')

        constant(self, 'target', target)
        constant(self, 'diffusion', diffusion)
        constant(self, 'lifetime', lifetime)
        constant(self, 'velocity', velocity)
        constant(self, 'vaxis', vaxis, index_dtype)
        constant(self, 'fluctuate', fluctuate, bool)
        if tshift is not None:
            constant(self, 'tshift', tshift)
        else:
            self.tshift = None
        if drtoa is not None:
            constant(self, 'drtoa', drtoa)
        else:
            self.drtoa = None

    def forward(self, time, charge, tail, head=None):
        '''
        Drift depos or steps.
        Requried Arguments:
        - time :: 1D tensor, the time of each step/depo in a shape of (npt, )
        - charge :: 1D tensor, the charge of each step/depo in a shape of (npt, )
        - tail :: 2D tensor in a shape of (npt, vdim) or 1D tensor in a shape
                  of (npt,), the position of depo when head is None, or end
                  point of each step when head is given.
        - head :: 2D tensor in a shape of (npt, vdim) or 1D tensor in a shape
                  of (npt,), the position of start point of each step.

        Note: the meaning of tail and head are swappable when they are given together.
              The method internally redefines the tail to be closet point to anode
              and the head to be farthest point to anode. It is worth noting the
              swap does not affect the input but gives the definition of dtail,
              dhead at the output.

        Returns tuple one larger than input args with sigma prepended.

        dsigma :: post-drift diffusion width at target plane.
        dtime :: post-drift time is the time for tail. Shifts may be applied.
        dcharge :: post-drift charge at target plane.
        dtail :: post-drift positions of depo or point closest to anode of each step.
        dhead :: post-drift positions of depo, or point farthest to anode of each step.
        '''

        if head is not None:
            tail, head = Drifter._ensure_tail_closer_to_anode(tail, head,
                                                     self.velocity, self.vaxis)
        # note, can in principle, drift with pare-existing sigmas.  But here we
        # assume point-like.
        dtail, dtime, dsigma, dcharge = drift(tail, velocity=self.velocity,
                                             diffusion=self.diffusion,
                                             lifetime=self.lifetime,
                                             target=self.target, times=time,
                                             vaxis=self.vaxis, charge=charge,
                                              fluctuate=self.fluctuate,
                                              tshift=self.tshift, drtoa=self.drtoa)
        if head is None:
            return (dsigma, dtime, dcharge, dtail)
        dhead = head + (dtail - tail)
        return (dsigma, dtime, dcharge, dtail, dhead)


    @staticmethod
    def _ensure_tail_closer_to_anode(tail, head, velocity, vaxis):
        """
        Ensures that the `tail` position is always closer to the anode than the `head`.

        This function reorders the `tail` and `head` tensors based on their positions along a specified axis (`vaxis`).
            The reordering is determined by the sign of `velocity`:
            - If `velocity > 0`, the function ensures `tail[:, vaxis] > head[:, vaxis]`, swapping elements where necessary.
            - If `velocity < 0`, it ensures `tail[:, vaxis] < head[:, vaxis]`.

        The swap operation is performed in-place for efficiency.

        Parameters:
            - tail (torch.Tensor): Tensor representing tail positions, either 1D (`(npt,)`) or 2D (`(npt, vdim)`).
            - head (torch.Tensor): Tensor representing head positions, either 1D (`(npt,)`) or 2D (`(npt, vdim)`).
            - velocity (float): Determines the direction of drift. Positive values prioritize larger `tail[:, vaxis]`, 
                            while negative values prioritize smaller ones.
            - vaxis (int): The index of the axis along which the positions are compared.

        Returns:
            - tuple (torch.Tensor, torch.Tensor): The reordered `tail` and `head` tensors.

        Raises:
            - ValueError: If `tail` and `head` do not have the same shape.

        Notes:
            - If the input tensors are 1D, they are temporarily expanded to 2D for processing and then restored to 1D.
        """

        if tail.shape != head.shape:
            raise ValueError(
                f"tail and head must have the same shape. Got tail: {tail.shape}, head: {head.shape}."
            )

        tail_new, head_new = tail.clone().detach(), head.clone().detach()

        # Handle 1D input by temporarily adding an extra dimension
        squeeze = tail.dim() == 1
        if squeeze:
            tail_new, head_new = tail_new.unsqueeze(1), head_new.unsqueeze(1)

        if vaxis >= tail_new.size(1):
            raise ValueError(
                f'Allow vaxis is only up to {tail.size(1)}.'
            )

        # Identify indices where swapping is needed
        mask = (tail_new[:, vaxis] < head_new[:, vaxis]) if velocity > 0 else (tail_new[:, vaxis] > head_new[:, vaxis])
        swap_idx = torch.nonzero(mask, as_tuple=True)[0]

        # Swap elements in-place
        tail_new[swap_idx], head_new[swap_idx] = head_new[swap_idx].clone(), tail_new[swap_idx].clone()

        # Restore original shape if necessary
        if squeeze:
            tail_new, head_new = tail_new.squeeze(1), head_new.squeeze(1)

        return tail_new, head_new


class Raster(nn.Module):
    '''
    Raster depos or steps.
    '''
    def __init__(self, velocity, grid_spacing, pdims=(1,2), tdim=-1, nsigma=3.0):
        '''
        - pdims :: the N-1 dimensions for transverse pitch indexing into input points.
        - tdim :: the 1 dimension for time/drift.
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

    def _transform(self, point, time):
        if point is None:
            return point
        nbatch = point.shape[0]
        ndims = len(self._pdims) + 1
        new_point = torch.zeros((nbatch, ndims), device=point.device)
        if self._pdims:
            for ind, pdim in enumerate(self._pdims):
                new_point[:,ind] = point[:,pdim]

        # FIXME: This is a bug! We need to set tail/head time dimension slightly
        # differently to preserve the step "angle" in the pitch-vs-time
        # dimensions.
        new_point[:,self._tdim] = time

        return new_point
        
    def forward(self, sigma, time, charge, tail, head=None):
        '''
        Raster the input depos, return block.

        Input drifted charge undergoes a transformation of the tensor dimensions
        via pdims and tdim.

        If head is None then tail is a depo point.  Otherwise the two make a step.
        '''
        tail = self._transform(tail, time)

        debug(f'grid:{tenstr(self.grid_spacing)} tail:{tenstr(tail)} sigma:{tenstr(sigma)} charge:{tenstr(charge)}')
        if head is None:        # depos, not steps
            rasters, offsets = raster_depos(self.grid_spacing, tail, sigma, charge, nsigma=self.nsigma)
            return Block(location = offsets, data=rasters)

        head = self._transform(head, time)
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
    
