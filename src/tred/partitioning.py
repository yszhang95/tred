#!/usr/bin/env python
'''
Functions for "partitioning" blocks.

Partitioning (here) de-interlaces a block into many smaller blocks, each
containing data from a common impact position.

Ignoring batch any drift/time dimension, a single transverse dimension may be
thought of as this tensor:

    [|0,1,2,3,4,5,6,7,9,|0,1,2,3,4,5,6,7,9,|....]

In this example, the super-grid spacing is 10 and the "|" marks the grid points
that are associated with super-grid points.  The numbers count the grid points
as impact positions relative to the super-grid points.

Partitioning this 1D tensor would generate a 1D collection of 1D tensors:

    [[0,0,...], [1,1,...], ..., [9,9,...]]

For 2 transverse dimensions this would generate a 2D collection of 2D tensors.  
'''

from itertools import product
from .util import to_tensor, to_tuple
from .blocking import Block
from .types import Tensor, IntTensor
from typing import Generator
import torch

def deinterlace(ten: Tensor, steps: IntTensor) -> Generator[Tensor, None, None]:
    '''
    Yield tensors that have been interlaced in ten at given step.

    - ten :: an N-dimensional interlaced tensor.

    - steps :: an intenger N-tensor giving the per-dimension step size of the interlacing.

    The shape of ten must be an integer multiple of steps

    A dimension with step of 1 effectively means no interlacing and each "lace"
    will have that dimension the same size as in ten.

    In general, each "lace" will be of shape ten.shape/steps.  The total number
    of "laces" yielded will be torch.prod(steps).
    '''
    if len(ten.shape) != len(steps):
        raise ValueError(f'dimensionality mismatch {len(ten.shape)} != {len(steps)}')

    if torch.any(torch.tensor(ten.shape, device=steps.device) % steps):
        raise ValueError(f'tensor of shape {ten.shape} not an integer multiple of {steps}')
        
    steps = to_tuple(steps)

    all_imps = [list(range(s)) for s in steps]
    for imps in product(*all_imps):
        slcs = [slice(imp,None,step) for imp,step in zip(imps,steps)]
        t = ten[slcs]
        # print(f'deinterlace: {t.device} {t.shape} {slcs}')
        yield t



def deinterlace_block(block: Block, spacing: Tensor, taxis: int = -1) -> Block:
    '''
    Iterate over impact de-interlace partitioning.

    - block :: an N-dimensional Block with data to be partitioned.  

    - spacing :: an intenger N-tensor giving the super-grid spacing.

    - taxis :: the element of spacing corresponding to the time/drift axis.
      Only non-taxis dimensions are partitioned.

    Yield element in collection of smaller blocks partitioned along non-taxis
    dimensions.

    Ignoring the batch dimension of block, a shape (n1,n2,n3) with taxis=-1 and
    spacing (m1,m2,m3) would generate m1*m2 smaller tensors each of shape
    (n1/m1, n2/m2, n3).  The taxis dimension is not partitioned.
    '''
    if block.vdim == 1:
        raise ValueError('partitioning 1D block not supported') # nonsensical
    if block.vdim != len(spacing):
        raise ValueError(f'block ({block.vdim}) and spacing ({len(spacing)} dimensionality differs')

    # The location for the yielded impact blocks
    scale = to_tensor(spacing)
    scale[taxis] = 1.0
    location = block.location // scale

    steps = to_tuple(spacing)

    all_imps = [list(range(s)) for s in steps]

    for imps in product(*all_imps):
        slcs = [slice(imp,None,step) for imp,step in zip(imps,steps)]
        slcs[taxis] = slice(0, None)
        data = block.data[:, *slcs]
        yield Block(location=location, data=data)

