#!/usr/bin/env python
'''
Some basic utility functions
'''
import sys
import subprocess
from pathlib import Path
from typing import Sequence, Iterator, Tuple

import functools
import warnings

import torch
from torch import Tensor

try:
    import magic
except ImportError:
    magic = None


import logging

log = logging.getLogger("tred")
debug = log.debug
warning = log.warning
info = log.info

def tenstr(ten):
    '''
    Return short string rep of a tensor
    '''
    s = to_tuple(ten.shape)
    return f'<tensor {s} {ten.device} {ten.dtype}>'

def setup_logging(log_output, log_level, log_format=None):

    try:
        level = int(log_level)      # try for number
    except ValueError:
        level = log_level.upper()   # else assume label
    log.setLevel(level)

    if log_format is None:
        log_format = '%(levelname)s %(message)s (%(filename)s:%(funcName)s)'
    log_formatter = logging.Formatter(log_format)

    def setup_handler(h):
        h.setLevel(level)
        h.setFormatter(log_formatter)
        log.addHandler(h)


    if not log_output:
        log_output = ["stderr"]
    for one in log_output:
        if one in ("stdout", "stderr"):
            setup_handler(logging.StreamHandler(getattr(sys, one)))
            continue
        setup_handler(logging.FileHandler(one))

    debug(f'logging to {log_output} at level {log_level}')


def mime_type(path):
    '''
    Return the canonical MIME content-type for file at path or None.
    '''
    path = Path(path)
    if not path.exists():
        return None

    if magic:
        return magic.from_file(path, mime=True)
    return subprocess.check_output(f"file --brief --mime-type {path}", shell=True).decode().strip()


def make_points(num, vdim, bb=None):
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
        bb = torch.tensor([0,1]*dims).reshape(-1,2)
        
    dimpts = list()
    for pmin,pmax in bb:
        uni = torch.rand(num, dtype=torch.float32)
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
    return torch.tensor(thing, device=device, dtype=dtype or torch.int32)


def slice_first(slc):
    return 0 if slc.start is None else slc.start

def slice_length(slc, tensor_length):
    # initial code: lamma3:8B.  fixing more bugs than there are lines of code: human.
    if slc.step is None or slc.step == 1:
        return min(slc.stop if slc.stop is not None else tensor_length, tensor_length) - (slice_first(slc) if slc.start is not None else 0)

    start = slice_first(slc)
    stop = min(slc.stop if slc.stop is not None else tensor_length, tensor_length)
    return max(0, 1+(stop - start) // abs(slc.step))



def getattr_first(attr, *things):
    '''
    Return the first .device found
    '''
    for thing in things:
        got = getattr(thing, attr, None)
        if got is not None:
            return got


def deprecated(reason):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapped
    return decorator


def iter_tensor_chunks(
    tensors: Sequence[Tensor],
    *,
    chunk_size: int | None = None,
    dim: int = 0,
) -> Iterator[Tuple[Tensor, ...]]:
    """
    Yield tuples of tensor views, chunked in sync along `dim`.

    Args:
        tensors: List/tuple of tensors to chunk together.
        chunk_size: Number of items per chunk (mutually exclusive with num_chunks).
        dim: Dimension along which to chunk (default 0).

    Yields:
        Tuple of tensor views, one view per input tensor, for each chunk.

    Notes:
        - Uses .narrow(...) so chunks are views (no data copy).
        - All tensors must have the same size along `dim`.
    """
    if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
        return
    if (chunk_size is None):
        raise ValueError("Specify exactly one of chunk_size.")
    if any(not isinstance(t, Tensor) for t in tensors):
        raise TypeError("All items must be torch.Tensors.")

    # Normalize dim (support negative dims)
    ref = tensors[0]
    D = ref.dim()
    dim = dim if dim >= 0 else D + dim
    if dim < 0 or dim >= D:
        raise ValueError(f"dim {dim} out of range for tensors with {D} dims.")

    # Ensure equal length along chunking dim
    n = ref.size(dim)
    for t in tensors[1:]:
        if t.size(dim) != n:
            raise ValueError("All tensors must match in size along the chunking dimension.")

    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            size = end - start
            yield tuple(t.narrow(dim, start, size) for t in tensors)
            start = end
