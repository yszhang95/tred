#!/usr/bin/env python
'''
Functions to deal with responses.

'''
import numpy
import torch

from .util import to_tensor, to_tuple

class Response:
    '''
    Bundle up tred response data.
    '''

    def __init__(self, current, spacing, start=0, axis=-1):
        '''
        Create response data.

        - current :: a tred response current float valued N-dim tensor.  
        - spacing :: a N-tuple giving grid spacing over N-1 spatial and 1 time dimensions.
        - start :: the absolute position along the nominal drift dimension of the response plane.
        - axis :: the drift time dimension, must be consistent with eg depo/step coordinate system.
        '''
        self.current = current
        if isinstance(spacing, torch.Tensor):
            self.spacing = spacing
        else:
            self.spacing = to_tensor(spacing, dtype=torch.float32)
        self.start = start
        self.axis = axis

# Move to util
def axis_last(ten, axis=-1):
    '''
    Return tensor transposed so that the given axis is the last.

    WARNING: This function transforms axes differently from `_transform` in `Drifter`.
             This function simply swap the axes. The other
             one move the axis to the last without changing the others' order.
    '''
    if axis == -1 or axis == len(ten.shape) - 1:
        return ten
    return torch.transpose(ten, axis, -1)


def quadrant_copy(raw, axis=-1, even=True):
    '''
    Return tensor with raw copied around each quadrant.

    - raw :: a tensor with first elements at the center and spanning the
             positive quadrant on the non-axis dimensions.

    - axis :: the drift time like dimension.

    - even :: set to True if raw[0,0,0] is just off the origin (even number of
      elements per pitch).  False if raw[0,0,0] is exactly at the origin (odd
      number symmetry).

    The result will have raw copied and rotated so that it is "centered".  See
    docs/response.org for details.
    '''
    if not even:
        # I think the odd case works same as even but after filling "full" we
        # strip off 1 "row" on the high-side of the first two dimensions.  This
        # needs more thought and debugging than I want to give right now.
        raise NotImplementedError('quadrant_copy is only implemented for an even number of elements per pitch')
    

    if len(raw.shape) != 3:
        raise TypeError(f'quadrant_copy operates only on 3D tensor, got {len(raw.shape)}')

    raw = axis_last(raw, axis)

    shape = torch.tensor(raw.shape, dtype=torch.int32) * 2
    shape[-1] = shape[-1] // 2
    full = torch.zeros(to_tuple(shape)).to(dtype=raw.dtype)

    h0 = raw.shape[0]
    h1 = raw.shape[1]

    #      0    1  2
    full[  :h0,   :h1, :] = raw
    full[h0:,     :h1, :] = raw.flip(dims=[0])
    full[  :h0, h1:,   :] = raw.flip(dims=[1])
    full[h0:,   h1:,   :] = raw.flip(dims=[0,1])

    return axis_last(full, axis)
    

def ndlarsim(npy_path):
    '''
    Load a response file from the original ND Lar simulation from a file named like:

    response_38_v2b_50ns_ndlar.npy

    And return a tred response object.
    '''
    raw = numpy.load(npy_path)
    if raw.shape != (45,45,6400):
        raise ValueError(f'unexpected shape {raw.shape} from {npy_path}')
    raw = torch.from_numpy(raw.astype(numpy.float32))
    return quadrant_copy(raw)

