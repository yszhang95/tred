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

    # The input `raw` tensor represents a quadrant located in the positive-positive corner.
    # Shifting the origin to the lower corner of the pixel means the new indices of input
    # must cover a larger positive index range.

    # positive-positive quadrant (original): no flip needed
    full[h0:  , h1:  , :] = raw
    # negative-positive quadrant: flip vertically (axis 0)
    full[  :h0, h1:  , :] = raw.flip(dims=[0])
    # positive-negative quadrant: flip horizontally (axis 1)
    full[h0:  ,   :h1, :] = raw.flip(dims=[1])
    # negative-negative quadrant: flip both axes
    full[  :h0,   :h1, :] = raw.flip(dims=[0,1])

    return axis_last(full, axis)


def ndlarsim(npy_path, nd_response_shape=None, nd_nimp=10):
    '''
    Load a response file from the original ND Lar simulation from a file named like:

    response_38_v2b_50ns_ndlar.npy

    And return a tred response object.

    Arguments:
        - npy_path :: Path to the response .npy file.
        - nd_response_shape :: Optional expected shape in pixel domain (Pxl, Pxl[, T]).
        - nd_nimp :: Number of impact positions per axis (default: 10).

    The input response is expected to be a quadrant of the full response, with a shape of (45, 45, 6400).
    It is aligned to the center of the collection pixel, rather than the lower corner.

    The resulting response is then shifted to align with the lower corner of the collection pixel.

    In addition to applying a shift, we transform the response to reflect an alternative explanation.
    The array at [0, 0] corresponds to the case where the electron is located at the lower corner of the pixel, generating current in the pixel it lands on.
    The array at [10, 0] corresponds to the case where the electron is at [0, 0], but the current is sensed in the pixel whose lower corner is at [10, 0].

    The shift and response shape are hard-coded. Use with caution.
    '''
    nd_response_shape = list([45, 45,]) if nd_response_shape is None else list(nd_response_shape) # 4.5 pixels; pixel is aligned to the center
    response_nimp = nd_nimp
    response_npxl = nd_response_shape[0]*2//response_nimp

    raw = numpy.load(npy_path)
    if raw.shape[0] != raw.shape[1]:
        raise ValueError(f"Number of pixels along each dimension must be equal. {raw.shape[:2]} is given.")
    if len(nd_response_shape) == 3:
        response_nt = nd_response_shape[-1]
    else:
        response_nt = raw.shape[-1]
        nd_response_shape.append(raw.shape[-1])
    nd_response_shape = tuple(nd_response_shape)

    if raw.shape != nd_response_shape:
        raise ValueError(f'unexpected shape {raw.shape} from {npy_path}')

    raw = torch.from_numpy(raw.astype(numpy.float32))
    full_response = quadrant_copy(raw).contiguous()
    response = full_response.view(response_npxl, response_nimp, response_npxl, response_nimp, response_nt)
    response = torch.flip(response, dims=(0, 2)).reshape(response_npxl*response_nimp, response_npxl*response_nimp, response_nt)

    return response.contiguous()

