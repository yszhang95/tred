#!/usr/bin/env python
'''
Some basic utility functions
'''
import subprocess
from pathlib import Path
import torch

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

