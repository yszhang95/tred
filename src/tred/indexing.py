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
    print(f'crop_batched:\n{offsets=}\n{inner=}\n{outer=}')
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
    

def chunk_location(offset, orig_shape, chunk_shape):
    '''
    Return locations of chunks made from a block.

    The `offset` is a 1-D integer N-tensor (or 2-D if batched) of index
    locations of a block (or blocks if batched).  Shape: (nbatch, N).

    The `orig_shape` is a tuple or 1-D integer N-tensor giving the block shape
    (common across all if batched).  Shape: (N,)

    The `chunk_shape` is a tuple or 1-D integer N-tensor giving the desired
    chunk shape.  Shape: (N,)

    Return location of chunks as N+1 (or 1+N+1 if batched) integer tensor giving
    the N-tensor offsets of the chunks.

    The Return Shape: [nbatch] + (orig_shape/chunk_shape) + [N]

    For example: given `chunk_shape==(2,3,4)` and `orig_shape==(6,3,8)` and
    `shape(offset)==(10, 3)` with 10 batches and N=3 dimension offsets gives a
    chunk multiple `M=(3,1,2)` spans the original shape.  The returned tensor
    would have shape `(10, 3,1,2, 3)` giving a 3-vector at each (3,1,2) chunk
    for each of 10 batches.

    This returns a tensor that corresponds to one returned by chunk().  The last
    dimension of size N returned by this function corresponds to the last
    N-dimensions (ie, the chunk dimensions) returned by chunk().

    To flatten the chunk-index dimension to get a flat batch of offset vectors:

    >>> coffs = chunk_location(offsets, orig_shape, chunk_shape)
    >>> batched_coffs = coffs.flatten(0, len(chunk_shape))
    '''
    if isinstance(orig_shape, torch.Tensor):
        orig_shape = tuple(orig_shape.tolist())
    if isinstance(chunk_shape, torch.Tensor):
        chunk_shape = tuple(chunk_shape.tolist())

    vdim = len(orig_shape)      # the "N" in the docstring
    if len(offset) == vdim:
        offset = offset[None,:] # Assure batched
    nbatches = offset.shape[0]

    grids = list()
    # per-dimension grids
    for o,c in zip(orig_shape, chunk_shape):
        r = torch.arange(0, o, c)
        #grids.append(r.to(dtype=torch.int32))
        grids.append(r)
    mg = torch.meshgrid(*grids, indexing='ij')

    off_mg = list()
    for idim in range(vdim):
        print(f'{idim=} {offset=}')
        off = offset[:,idim]    # one space dim offsets along batch dim
        for count in range(vdim):
            off = torch.unsqueeze(off, -1) # match space dims
        mgd = mg[idim][None,:]  # add batch dim

        prod = torch.unsqueeze(mgd*off, -1) # dimension 1+N+1
        off_mg.append(prod)
    return torch.cat(off_mg, dim=-1)
    

def chunk(ten, shape):
    '''
    Return content of tensor ten in chunks of shape.

    The `shape` is 1-D integer N-tensor giving the desired shape of chunks.

    The `ten` may have N or 1+N dimensions if batched.  The size of the N
    dimensions must be a multiple of `shape`, that is `M = ten.shape/shape` is
    an integer tensor (written assuming ten is unbatched).

    The dimension of the returned chunked tensor is N+N or 1+N+N if batched.

    The first N dimensions (ignoring a possible first batched dimension) span
    the chunks.  The size of these dimensions is `M`, the integer multiple
    defined above.  The order of these dimensions follows that of `shape` and
    `ten`.  Eg, the first dimension of N corresponds to the first dimension of
    `ten` and `shape`.

    The last N dimensions span individual elements of a chunk and these
    dimensions have size and ordering as given by `shape`.

    For example: `shape(shape)==(2,3,4)` and `shape(ten)==(10, 6,3,8)` with 10
    batches implies the multiple `M=(3,1,2)`.  The returned chunked tensor would
    have shape `(10, 3,1,2, 2,3,4)`.

    Note, to flatten the chunk-index dimensions to get a flat batch of chunks:

    >>> rs = chunk(ten, shape).flatten(0, len(shape))

    With the example above, this gives a tensor of shape `(nbatch*3*1*2, 2,3,4)`

    See also chunk_location()

    '''
    vdim = len(shape)           # the "N" in the docstring
    if len(ten.shape) == vdim:
        ten = ten[None,:]       # assure batched

    eshape = torch.tensor(ten.shape[1:], dtype=torch.int32)
    if isinstance(shape, torch.Tensor):
        cshape = shape
    else:
        cshape = torch.tensor(shape, dtype=torch.int32)
    fshape = eshape/cshape
    mshape = fshape.to(torch.int32)
    if not torch.all(mshape == fshape):
        raise ValueError(f'non batch dimensions of tensor of shape {ten.shape} are not integer multiples of chunk shape {shape}')

    # See test_indexing.py's test_chunking_*() for demystifying this.  We
    # construct the arg lists for reshape+permute+reshape in a
    # N-dimension-independent manner.

    # The first reshape creates the axes.
    rargs1 = tuple([-1] + torch.vstack((mshape, cshape)).T.flatten().tolist())
    # The transpose/permute brings the axes into correct order.
    pargs = tuple([0] + list(range(1,2*vdim,2)) + list(range(2,2*vdim+1,2)))
    # The final reshape
    rargs2 = tuple([-1] + mshape.tolist() + cshape.tolist())

    return ten.reshape(*rargs1).permute(*pargs).reshape(*rargs2)
