#!/usr/bin/env python
'''
Support for DFT-based convolutions.

'''

from .util import to_tuple, to_tensor
from .types import Tensor, Shape
import torch
from torch.nn.functional import pad


def dft_shape(ten, kern):
    '''
    Return the shape required to convolve tensor ten with kernel.

    The tensor ten may be batched.  The returned shape does not include a batch
    dimension.
    '''
    if len(ten.shape) == len(kern.shape):
        tshape = to_tensor(ten.shape)
    else:
        tshape = to_tensor(ten.shape[1:])
    kshape = to_tensor(kern.shape)
    return to_tuple(tshape + kshape - 1)


def zero_pad(ten : Tensor, shape: Shape|None = None) -> Tensor:
    '''
    Zero-pad tensor ten to be of given shape.

    If the number of tensor dimensions of ten is one more than the length of
    shape then an extra first dimension is assumed to run along a batch.

    Zero-padding is applied to the high-side of each non-batch dimension.

    For padding dimensions that are "centered" consider to bracket a call to
    zero_pad() with calls to torch.roll().  A positive shift, then the pad
    followed by a negative shift.
    '''
    batched = True
    if len(shape) == len(ten.shape):
        batched = False
        ten = torch.unsqueeze(ten, 0) # add batch dim.

    have = to_tensor(ten.shape[1:], dtype=torch.int32)
    want = to_tensor(shape, dtype=torch.int32)
    zzzz = torch.zeros_like(want)

    diff = want - have
    padding = torch.vstack((diff, zzzz)).T.flatten().tolist()
    padding.reverse()
    padded = pad(ten, tuple(padding))
    if not batched:
        padded = padded[0]
    return padded
        

def convolve(ten, kernel):
    '''
    Return the convolution of tensor ten with kernel.

    Both input tensors are real-valued interval domain.

    The ten tensor may be batched, but not the kernel.
    '''

    
