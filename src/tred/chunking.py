#!/usr/bin/env python
'''
tred.chunking provides functions to chunk N-dimensional tensors.

See also tred.sparse for related functions that depend on the idea of a
super-grid.
'''


from .blocking import Block
from .util import to_tuple
from .types import Shape, Tensor

from collections import defaultdict
import torch



def location(envelope: Block, chunk_shape:Shape):
    '''
    Translate envelope locations to those for given chunk size.

    See `tred.chunking.content()`.

    The `envelope` provides a block with a shape that is an integer multiple of
    `chunk_shape`.

    Return the locations of the chunks in the envelope as integer,
    1+N+1-dimension tensor.  In detail, the shape of the returned tensor is:

        [1] + M + [N]

    Where the first is the batch dimension, M is the "integer-multiple vector"
    given by the ratio of the envelope and chunk shapes and N is the number of
    spatial dimensions.

    For example: given `chunk_shape==(2,3,4)` and `envelope.shape==(6,3,8)` and
    `envelope.location.shape==(10, 3)` with 10 batches and N=3 spatial
    dimensions gives an integer-multiple vector of `M=(3,1,2)`.  The returned
    location tensor would have shape `(10, 3,1,2, 3)`.  Last dimension gives
    3-vectors locating each of (3,1,2) chunks for each of 10 batches.

    This location tensor corresponds to a tensor returned by
    `chunking.content()` given the same arguments.  The last 1 dimension of size
    N returned by this function corresponds to the last N dimensions of the
    tensor returned by the other of the other.

    To flatten the "chunk dimensions" (eg (3,2,1) in the above example) in order
    to get a flat batch of location vectors one may do:

    >>> clocs = chunking.location(envelope, chunk_shape)
    >>> clocs = clocs.flatten(0, len(chunk_shape))
    '''
    if isinstance(envelope, Tensor):
        raise TypeError(f'chunking.location() requires a Block, got {type(envelope)}')

    orig_shape = to_tuple(envelope.shape)
    chunk_shape = to_tuple(chunk_shape)

    vdim = len(orig_shape)      # the "N" in the docstring

    grids = list()
    # per-dimension grids
    for o,c in zip(orig_shape, chunk_shape):
        r = torch.arange(start=0, end=o, step=c)
        grids.append(r)
    mg = tuple([one.to(device=envelope.device) for one in torch.meshgrid(*grids, indexing='ij')])

    off_mg = list()
    for idim in range(vdim):
        off = envelope.location[:,idim]    # one space dim offsets along batch dim
        for count in range(vdim):
            off = torch.unsqueeze(off, -1) # match space dims
        mgd = mg[idim][None,:]  # add batch dim

        prod = torch.unsqueeze(mgd+off, -1) # dimension 1+N+1
        off_mg.append(prod)
    return torch.cat(off_mg, dim=-1)


def content(envelope, chunk_shape):
    '''
    Return a reshaped tensor from envelope.data that has chunk-level
    dimensions sized as chunk_shape.

    The `chunk_shape` is 1-D integer N-tensor giving the desired shape of chunks.

    The shape of the spatial (non-batch) dimensions of `envelope` MUST be
    integer multiples of `chunk_shape`.

    The dimension of the returned chunked tensor is 1+N+N.

    The first N dimensions (ignoring a possible first batched dimension) span
    the chunks.  The size of these dimensions is `M`, the integer multiple
    defined above.  The order of these dimensions follows that of `chunk_shape` and
    `envelope`.  Eg, the first dimension of N corresponds to the first dimension of
    `envelope` and `chunk_shape`.

    The last N dimensions span individual elements of a chunk and these
    dimensions have size and ordering as given by `chunk_shape`.

    For example: `chunk_shape==(2,3,4)` and `shape(envelope)==(10, 6,3,8)` with 10
    batches implies the multiple `M=(3,1,2)`.  The returned chunked tensor would
    have shape `(10, 3,1,2, 2,3,4)`.

    Note, to flatten the chunk-index dimensions to get a flat batch of chunks:

    >>> rs = chunk(envelope, chunk_shape).flatten(0, len(chunk_shape))

    With the example above, this gives a tensor of shape `(nbatch*3*1*2, 2,3,4)`

    See also location()
    '''
    if isinstance(envelope, Tensor):
        raise TypeError(f'chunking.content() requires a Block, got {type(envelope)}')

    chunk_shape = envelope.to_tensor(chunk_shape)

    vdim = len(chunk_shape)           # the "N" in the docstring

    fshape = envelope.shape/chunk_shape
    mshape = fshape.to(torch.int32)
    if not torch.all(mshape == fshape):
        raise ValueError(f'chunking.content(): spatial dimension size mismatch: {envelope.shape=}  not integer multiple of {chunk_shape=}')

    # We construct the arg lists for reshape+permute+reshape in a
    # N-dimension-independent manner.  See test_indexing.py's test_chunking_*()
    # for demystifying this.

    # The first reshape creates the axes.
    rargs1 = tuple([-1] + torch.vstack((mshape, chunk_shape)).T.flatten().tolist())
    # The transpose/permute brings the axes into correct order.
    pargs = tuple([0] + list(range(1,2*vdim,2)) + list(range(2,2*vdim+1,2)))
    # The final reshape
    rargs2 = tuple([-1] + mshape.tolist() + chunk_shape.tolist())

    return envelope.data.reshape(*rargs1).permute(*pargs).reshape(*rargs2)


def accumulate(chunk : Block) -> Block:
    '''
    Sum chunks at the same location, skipping any that are empty
    '''
    loc = chunk.location
    dat = chunk.data
    # Step 1: Get unique keys and their inverse index
    unique_locs, inv = torch.unique(loc, dim=0, return_inverse=True)

    # Step 2: Prepare output tensor
    summed = torch.zeros((unique_locs.shape[0], *dat.shape[1:]), dtype=dat.dtype).to(dat.device)

    # Step 3: Use index_add to accumulate
    summed = summed.index_add(dim=0, index=inv, source=dat, alpha=1)

    # Step 4: Drop zero contents
    indices = torch.where(summed.sum(dim=[i+1 for i in range(chunk.vdim)]))

    return Block(location=unique_locs[indices], data=summed[indices])

