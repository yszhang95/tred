#!/usr/bin/env python
'''
Functions to help calculate indices
'''

import torch

def shape_meshgrid(shape):
    '''
    Return a meshgrid of indices spanning the shape.

    >>> shape_meshgrid((2,3))
    ... (tensor([[0, 0, 0], [1, 1, 1]]),
         tensor([[0, 1, 2], [0, 1, 2]]))
    '''
    vdim = len(shape)
    odims = [torch.arange(o) for o in shape]
    return torch.meshgrid(*odims, indexing='ij')

    
def crop(offset, inner, outer):
    '''
    Return the indices of elements spanning the inner N-dimensional box
    shape located at the offset inside an outer box shape.

    All three inputs are 1D N-tensors.

    Indices are returned as 1D tensor and correspond to the elements of the
    flatten()'ing of the outer tensor.
    '''

    vdim = len(offset)

    odims = [torch.arange(o) for o in outer]
    omesh = torch.meshgrid(*odims, indexing='ij')

    indices = torch.ones(outer.tolist())
    for dim in torch.arange(vdim):
        indices[omesh[dim] < offset[dim]] = 0
        indices[omesh[dim] >= offset[dim] + inner[dim]] = 0

    return torch.nonzero(indices.flatten(), as_tuple=True)[0]

    

def crop_batched(offsets, inner, outer):
    '''
    Batched N-dimensional version of crop().

    - offsets :: 2 dimensional batched N-vector of location of inner shapes in the outer shape.
    - inners :: 2-dimensional batched or 1-dimensional common N-vector of inner shapes.
    - outer :: 1-dimensional common N-vector of outer shape.

    The input offsets and inners are 2D batched N-tensors.  The omesh may be a
    meshgrid (tuple of tensors) spanning the "outer" shape of crop() or it may
    be the outer shape itself (tuple or N-tensor).  The outer shape is common to
    all inners.
    '''
    if isinstance(outer, torch.Tensor):
        outer = tuple(outer.tolist())

    omesh = shape_meshgrid(outer)
    vdim = len(omesh)

    boff = omesh[0].numel()
    nbat = offsets.shape[0]

    batched_inner = False
    if len(inner.shape) > 1:
        batched_inner = True

    inds = list()
    for bind in range(nbat):
        keep = torch.ones(outer)

        offset = offsets[bind]
        if batched_inner:
            shape = inner[bind]
        else:
            shape = inner

        for dim in torch.arange(vdim):
            keep[omesh[dim] <  offset[dim]] = 0
            keep[omesh[dim] >= offset[dim] + shape[dim]] = 0
        
        ind = bind*nbat + torch.nonzero(keep.flatten(), as_tuple=True)[0]
        inds.append(ind)
    return torch.hstack(inds)


def union_bounds(shape, offsets):
    '''
    Return min index and shape that bounds the shape placed at many offsets.
    '''
    pmin = torch.min(offsets, dim=0)
    pmax = torch.max(torch.tensor(shape)+offsets, dim=0)
    return (pmin.values, (pmax.values-pmin.values).to(dtype=pmin.values.dtype))
    
