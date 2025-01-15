#!/usr/bin/env python
'''
Some basic utility functions
'''
import subprocess
from pathlib import Path
import torch
from torch import Tensor

try:
    import magic
except ImportError:
    magic = None


def mime_type(path):
    '''
    Return the canonical MIME content-type for file at path or None.
    '''
    path = Path(path)
    if not path.exists():
        return None

    if magic:
        return magic.from_file(path, mime=True)
    return subprocess.check_output(f"file --brief --mime-type {file}", shell=True).decode().strip()


def make_points(num, vdim, bb=None, device='cpu'):
    '''
    Return points as tensor shape (nums, vdim).

    - num :: number of points
    - vdim :: the vector-dimension (length) of the point vectors

    Points are generated uniformly random in a rectangular box of
    vector-dimension vdim defined by bounding box (bb) which has shape (2,vdim)
    and gives the points on two extreme corners.  If bb is None, the unit square
    with bounds [0,1) on each axis is generated.

    '''
    if bb is None:
        bb = torch.tensor([0,1]*dims, device=device).reshape(-1,2)
        
    dimpts = list()
    for pmin,pmax in bb:
        uni = torch.rand(num, dtype=torch.float32, device=device)
        dimpts.append( uni*(pmax-pmin)+pmin )
    return torch.vstack(dimpts).T

def to_tuple(thing):
    '''
    Try hard to convert a thing into a tuple.
    '''
    if isinstance(thing, tuple):
        return thing
    if isinstance(thing, torch.Tensor):
        thing = thing.tolist()
    else:
        thing = list(thing)
    return tuple(thing)

def to_tensor(thing, dtype=None, device=None):
    '''
    Try hard to convert thing into a torch.Tensor of given dtype and device
    '''
    if isinstance(thing, torch.Tensor):
        dtype = dtype or thing.dtype
        device = device or thing.device
        return thing.to(device=device, dtype=dtype)
    return torch.tensor(thing, device=device or 'cpu', dtype=dtype or torch.int32)

def slice_first(slc):
    return 0 if slc.start is None else slc.start

def slice_length(slc, tensor_length):
    if slc.step is None or slc.step == 1:
        return min(slc.stop if slc.stop is not None else tensor_length, tensor_length) - (slice_first(slc) if slc.start is not None else 0)

    start = slice_first(slc)
    stop = min(slc.stop if slc.stop is not None else tensor_length, tensor_length)
    # length = ((stop - start) // abs(slice.step)) + 1
    # length = min(stop, tensor_length) - max(start, 0)
    # return min(length, tensor-length - 
    return max(0, 1+(stop - start) // abs(slc.step))

