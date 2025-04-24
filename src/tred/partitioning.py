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
from typing import Generator, Tuple
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

def deinterlace_pairs(ten: Tensor, steps: IntTensor, pair_axis: int = 0) -> Generator[Tuple[Tensor, Tensor], None, None]:
    '''
    Yield tensors in pairs that have been interlaced in `ten` at given step.

    This function is similar to `deinterlace`, but it groups slices into symmetric pairs
    along the specified `pair_axis`.

    The symmetric pairing matches positions along `pair_axis` in a mirrored fashion:
    (0, N-1), (1, N-2), ..., where N is the number of interleaved positions along that axis
    (given by `steps[pair_axis]`).

    The slicing pattern follows a Cartesian product over all axes except `pair_axis`,
    which is handled last to enforce symmetric pairing.

    - pair_axis :: The axis along which a pair is generated.
    '''
    if len(ten.shape) != len(steps):
        raise ValueError(f'dimensionality mismatch {len(ten.shape)} != {len(steps)}')

    if torch.any(torch.tensor(ten.shape, device=steps.device) % steps):
        raise ValueError(f'tensor of shape {ten.shape} not an integer multiple of {steps}')

    steps = to_tuple(steps)

    assert steps[pair_axis] % 2 == 0, f'Number of impact positions along axis'\
        f' {pair_axis} is {steps[pair_axis]}, which should be even'

    pair_axis = len(steps) + pair_axis if pair_axis < 0 else pair_axis

    all_imps = [list(range(s)) for i, s in enumerate(steps) if i != pair_axis]

    for imps in product(*all_imps):
        for j in range(steps[pair_axis]//2):
            imps1 = [imps[i] for i in range(len(imps))]
            imps2 = [imps[i] for i in range(len(imps))]
            imps1.insert(pair_axis, j)
            imps2.insert(pair_axis, steps[pair_axis]-1-j)
            slcs1 = [slice(imp,None,step) for imp,step in zip(imps1,steps)]
            slcs2 = [slice(imp,None,step) for imp,step in zip(imps2,steps)]

            yield (ten[slcs1], ten[slcs2])

# #optimized
# def deinterlace_pairs(ten: Tensor, steps: IntTensor, pair_axis: int = 0) -> list:
#     '''
#     Returns batched tensors in pairs that have been interlaced in `ten` at given step.
    
#     This optimized version returns all pairs at once as a list instead of yielding them
#     one by one, allowing for more efficient GPU utilization.
    
#     - ten :: The tensor to deinterlace
#     - steps :: The step sizes for each dimension
#     - pair_axis :: The axis along which to create symmetric pairs
    
#     Returns a list of tuples, each containing a pair of tensors.
#     '''
#     if len(ten.shape) != len(steps):
#         raise ValueError(f'dimensionality mismatch {len(ten.shape)} != {len(steps)}')

#     if torch.any(torch.tensor(ten.shape, device=steps.device) % steps):
#         raise ValueError(f'tensor of shape {ten.shape} not an integer multiple of {steps}')

#     steps = to_tuple(steps)
#     device = ten.device

#     assert steps[pair_axis] % 2 == 0, f'Number of impact positions along axis'\
#         f' {pair_axis} is {steps[pair_axis]}, which should be even'

#     # Normalize negative pair_axis
#     pair_axis = len(steps) + pair_axis if pair_axis < 0 else pair_axis
    
#     # Create indices for all dimensions except pair_axis
#     dim_ranges = []
#     for i, s in enumerate(steps):
#         if i != pair_axis:
#             dim_ranges.append(torch.arange(s, device=device))
    
#     # Use meshgrid to create the Cartesian product of all other dimensions
#     # This replaces the itertools.product call
#     if len(dim_ranges) > 0:
#         meshes = torch.meshgrid(*dim_ranges, indexing='ij')
#         # Flatten the meshgrid results
#         grid_indices = [mesh.flatten() for mesh in meshes]
#     else:
#         # Handle case with only one dimension
#         grid_indices = []
    
#     # Pre-allocate all result pairs
#     half_pairs = steps[pair_axis] // 2
#     other_dims_count = 1
#     for i, s in enumerate(steps):
#         if i != pair_axis:
#             other_dims_count *= s
    
#     result_pairs = []
    
#     # Create all slices at once for each dimension except pair_axis
#     all_slices = {}
#     for dim_idx, (step, indices) in enumerate(zip([s for i, s in enumerate(steps) if i != pair_axis], 
#                                                  grid_indices)):
#         real_dim = dim_idx if dim_idx < pair_axis else dim_idx + 1
#         all_slices[real_dim] = [slice(idx.item(), None, step) for idx in indices]
    
#     # Batch process all pairs
#     for idx in range(other_dims_count):
#         # Get the slice for each dimension except pair_axis
#         for j in range(half_pairs):
#             # Create the slices for this specific combination
#             slcs1 = []
#             slcs2 = []
            
#             for dim in range(len(steps)):
#                 if dim == pair_axis:
#                     slcs1.append(slice(j, None, steps[dim]))
#                     slcs2.append(slice(steps[pair_axis]-1-j, None, steps[dim]))
#                 else:
#                     # Find the correct dimension index in all_slices
#                     dim_idx = dim if dim < pair_axis else dim - 1
#                     slc = all_slices[dim][idx % (len(all_slices[dim]))]
#                     idx = idx // len(all_slices[dim]) if len(all_slices[dim]) > 0 else idx
#                     slcs1.append(slc)
#                     slcs2.append(slc)
            
#             # Get the tensor slices
#             result_pairs.append((ten[slcs1], ten[slcs2]))
    
#     return result_pairs



# Not validated and not referenced elsewhere.
#
# The function seems to work under the assumption, the block is at the corner of the super grid. Then deinterlaced partitions are at the
# same position on the super grid (same indices). Then the returned location make senses. Function interlaced in tred.convo supports this opinion.
#
# Given a block (let us ignore batch dimension) at (1,2) or (2,2). The shape of the block is (4, 4). And we ignore taxis now.
# The step size is set to (2, 2). I expect returned data are data[[[0,2], [0,1,2,3]]] and data[[[1,3], [0,1,2,3]]].
# The returned location is at (1,2) // [2,2] -> [0,1]. I do not understand the meaning.
# When loation is at (2,2), then the resulted location [1,1] implies all partitions are at the index [1,1] of the super-grid.
def deinterlace_block(block: Block, spacing: Tensor, taxis: int = -1) -> Generator[Block, None, None]:
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

