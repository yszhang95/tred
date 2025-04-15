#!/usr/bin/env python
'''
Support for block-sparse binned (BSB) data.

See docs/concepts.org for a description of the concepts and their terms.

See tred.bsb for a class to add value to these functions.

'''


import torch

from .indexing import crop_batched
from . import chunking
from .util import to_tensor, to_tuple
from .blocking import Block
from .types import IntTensor, Shape, Tensor

offset_dtype = torch.int64
MAX_OFFSET = torch.iinfo(offset_dtype).max

# fixme: Does this REALLY deserve to be a class?
class SGrid:
    '''
    A super-grid is a subset of points with a given spacing selected from an
    underlying N-dimensional grid with unit spacing.
    '''

    spacing: IntTensor
    """1D integer N-vector giving number of underlying grid points per super-point on each dimension"""


    def __init__(self, spacing: IntTensor):
        '''
        Create a super-grid with given spacing.
        '''
        self.spacing = to_tensor(spacing)


    @property
    def vdim(self):
        '''
        The number of dimensions spanned by the grid.
        '''
        return len(self.spacing)


    def to_tensor(self, thing) -> Tensor:
        '''
        Return tensor assured to be on same device as spacing
        '''
        return to_tensor(thing, device=self.spacing.device)


    def spoint(self, gpoint:IntTensor, goffset:IntTensor|None=None) -> IntTensor:
        '''
        Return the bin-grid index vector that would contain the
        grid-point index vector in the bin's half-open extent.

        :param gpoint: point-grid index vector.
        :return: bin-grid index vector.

        If input is a 2D tensor, the first dimension runs over batches and the
        output is also a 2D batched tensor.

        Inverse of `gpoint`.
        '''
        if goffset is None:
            return self.to_tensor(gpoint) // self.spacing
        return (self.to_tensor(gpoint) - self.to_tensor(goffset)) // self.spacing

    def gpoint(self, spoint: IntTensor) -> IntTensor:
        '''
        Return the bin's grid index (bin's lower corner).

        :param spoint: bin-grid index vector.
        :return: point-grid index vector.

        If input is a 2D tensor, the first dimension runs over batches and the
        output is also a 2D batched tensor.

        Inverse of `spoint`.
        '''
        return self.to_tensor(spoint) * self.spacing
        

    def envelope(self, pbounds: Block) -> Block:
        '''
        Return the smallest block with volumes aligned to the super-grid
        and that contains volumes given by pbounds.

        Note, the Block.data of the returned is not defined.
        '''
        # fixme: can this method be made to work for batched shape?

        # Find the minimum bin span that contains the block.
        # Find the location of the bin containing the lowest point.
        minpts = self.gpoint(self.spoint(pbounds.location)) 
        # Find the location of the bin one past the bin containing the highest point.
        maxpts = self.gpoint(self.spoint(pbounds.location+pbounds.shape-1)+1)
        shapes = maxpts - minpts
        maxshape = torch.max(shapes, dim=0).values
        return Block(minpts, shape=maxshape)


def fill_envelope(envelope: Block, block: Block) -> Block:
    '''
    Fill envelope with block.

    If envelop lacks a .data attribute it will be allocated.
    '''
    if not isinstance(envelope, Block):
        raise TypeError(f'sparse.fill_envelope() envelope must be Block got {type(envelope)}')
    if not isinstance(block, Block):
        raise TypeError(f'sparse.fill_envelope() block must be Block got {type(block)}')

    if not hasattr(envelope, "data"):
        envelope.set_data(torch.zeros(envelope.size(),
                                      dtype=block.data.dtype, device=block.data.device))

    if len(block.shape) != len(envelope.shape):
        raise ValueError(f'fill_envelope shape mismatch with block {block.shape} and envelope {envelope.shape}')

    # location of block inside the envelope.
    offset = block.location - envelope.location
    if not torch.all(offset >= 0):
        raise ValueError(f'fill_envelope negative locations of block {block.shape} in envelope {envelope.shape}')

    inds = crop_batched(offset, block.shape, envelope.shape)
    envelope.data.flatten()[inds] = block.data.flatten()
    return envelope


def reshape_envelope(envelope: Block, chunk_shape:Shape):
    '''
    Return a block which reshapes the envelope to given chunk size.

    The location of the reshaped block reflects this chunking.
    '''

    loc = chunking.location(envelope, chunk_shape)
    dat = chunking.content(envelope, chunk_shape)

    vdim = len(chunk_shape)
    return Block( location=loc.flatten(0, vdim), data=dat.flatten(0, vdim) )


# def block_chunk(sgrid: SGrid, block: Block) -> Block:
#     '''
#     Break up block into aligned chunks.
#     '''
#     env = sgrid.envelope(block)
#     fill_envelope(env, block)
#     assert env.data is not None
#     return reshape_envelope(env, sgrid.spacing)



def chunkify(block: Block, shape: IntTensor) -> Block:
    '''
    Chunk the block into a new one with volume dimensions of given shape.
    '''
    sgrid = SGrid(shape)
    envelope = sgrid.envelope(block)
    fill_envelope(envelope, block)
    return reshape_envelope(envelope, sgrid.spacing)

def flat_index(ten, stride):
    if len(ten.shape) !=2:
        raise ValueError(f'sparse.flat_index() ten must be 2D tensor, but {len(ten.shape)} is given.')
    if ten.dtype != offset_dtype:
        raise ValueError(f'sparse.flat_index() ten must be in {offset_dtype}, but {ten.dtype} is given.')
    if stride.dtype != stride.dtype != offset_dtype:
        raise ValueError(f'sparse.flat_index() stride must be in {offset_dtype}, but {stride.dtype} is given.')
    if ten.size(1) != stride.numel():
        raise ValueError(f'Number of components in ten must match stride. stride length is {stride.numel()} while n components of ten is {ten.size(1)}')

    return torch.sum(ten * stride[None,...], dim=1)

def unflat_index(index, stride):
    if index.dim() != 1:
        raise ValueError(f'sparse.unflat_index() index must be 1D tensor, but {index.dim()}D is given.')
    if index.dtype != torch.int32 and index.dtype != torch.int64:
        raise ValueError(f'sparse.unflat_index() index must be in int32 or int64, but {index.dtype} is given.')
    if stride.dtype != torch.int32 and stride.dtype != torch.int64:
        raise ValueError(f'sparse.unflat_index() stride must be in int32 or int64, but {stride.dtype} is given.')

    coord = []
    rem = index
    for s in stride:
        coord.append(rem // s)
        rem = rem % s
    return torch.stack(coord, dim=1)

def calculate_strides(shape: tuple|list|Tensor):
    if not isinstance(shape, Tensor):
        shape = torch.tensor(shape, dtype=offset_dtype)
    if shape.ndim != 1:
        raise ValueError(f"sparse.calculate_strides must be in 1D, but {shape.ndim} is given")
    strides = [1,]
    for s in torch.flip(shape, dims=(0,)):
        strides.append(strides[-1] * s)
    return strides[::-1]


def chunkify2(block: Block, shape: IntTensor) -> Block:
    '''
    Chunk the block into a new one with volume dimensions of given shape.
    '''

    sgrid = SGrid(shape)
    if not isinstance(block, Block):
        raise TypeError(f'sparse.chunkify2() block must be Block got {type(block)}')
    envelope = sgrid.envelope(block)

    minpts = envelope.location

    inner = block.shape
    outer = envelope.shape
    nbatches = block.nbatches
    locs = block.location - minpts
    cshape = sgrid.spacing.to(locs.device)
    ndim = int(inner.numel())

    ldevice = block.location.device
    ddevice = block.data.device

    # location of block inside the envelope.
    outer_strides = torch.tensor(calculate_strides(outer), device=ldevice)
    inner_strides = torch.tensor(calculate_strides(inner), device=ldevice)
    chunk_strides = torch.tensor(calculate_strides(cshape), device=ldevice)
    ind_strides = torch.tensor(calculate_strides(outer // cshape), device=ldevice)

    if torch.any(outer % cshape):
        raise ValueError("chunk shape must be dividable by SGrid.sgrid")

    local_grid = torch.cartesian_prod(*[torch.arange(sz, device=ldevice) for sz in inner])  # [prod(inner), ndim]

    # flat according to outer // cshape
    tile_coords  = (locs[:, None, :] + local_grid[None, :, :]) // cshape     # [nbatches, prod(inner), ndim]
    tile_coords = flat_index(tile_coords.view(-1, ndim), ind_strides[1:]) # [nbatches * prod(inner),]
    tile_coords = tile_coords.view(nbatches, local_grid.size(0))
    bidx = torch.arange(0, nbatches*ind_strides[0]-1, ind_strides[0], dtype=offset_dtype, device=ldevice) # (nbatches,)
    tild_idx = (bidx[:,None] + tile_coords).view(-1) # (nbatches * prod(inner))

    tile_keys, reverse_indices = torch.unique(tild_idx, return_inverse=True, sorted=True)

    locs_new = unflat_index(tile_keys, ind_strides)
    locs_new[:,1:] = locs_new[:,1:] * cshape[None,:] 
    _, reverse_bidx = torch.unique(locs_new[:,0], sorted=True, return_inverse=True)
    locs_new = locs_new[:,1:] + minpts[reverse_bidx]

    # flat according to cshape
    local_offsets = (locs[:, None, :] + local_grid[None, :, :]) % cshape     # [nbatches, prod(inner), ndim]
    local_offsets = local_offsets.view(-1, ndim)
    local_offsets = flat_index(local_offsets, chunk_strides[1:]) # [nbatches * prod(inner),]

    # FIXME: CUDA-CPU SYNC
    data = torch.zeros((tile_keys.size(0),) + tuple(cshape.tolist()), dtype=block.data.dtype, device=ddevice)

    tile_indices = reverse_indices.view(-1)                     # [nbatches * prod(inner)]
    values = block.data.flatten()                               # [nbatches * prod(inner)]

    index = [tile_indices, local_offsets]
    data.view(-1, chunk_strides[0])[tuple(index)] = values

    return Block(location=locs_new, data=data)


def index_chunks(sgrid: SGrid, chunk: Block) -> Block:
    '''
    Index the chunks over space by bin.

    Chunks are assumed to be non-overlapping.  See chunking.accumulate().

    This returns a Block with a single batch and is defined on the super-grid.
    Each chunk is mapped to one element in the N-dimensional data which in turn
    holds the batch index of the chunk.  Elements with negative value indicate
    that no chunk exists.

    The returned location is on indices of the super-grid, i.e., from `sgrid.spoint`.

    The returned index block allows round trip navigation between a chunk and
    its neighbors.  For example:

    >>> ic = index_chunks(sgrid, chunk)
    >>> ic_origin = ic.location[0]
    >>> ic_index = ic.data[0]

    Round trip through the index

    >>> s_index = sgrid.spoint(chunk.location, sgrid.gpoint(ic_origin)).T
    >>> b_index = ic_index[s_index.tolist()]
    >>> assert torch.all(b_index == torch.arange(chunk.nbatches))

    Find nearest neighbor above a chunk along dimension zero.

    >>> nn = torch.zeros_like(ic_origin)
    >>> nn[0] = 1
    >>> nn += s_index
    >>> # fixme: we ignore overflowing s_index bounds
    >>> print("we have this many NN's:", torch.sum(ic_origin[nn] >= 0))

    '''

    # 1+1-dimension, shape (Nbatch, N)
    super_location = sgrid.spoint(chunk.location)
    # 1-D, shape(N)
    minp = super_location.min(dim=0).values
    maxp = super_location.max(dim=0).values
    shape = to_tuple(maxp - minp + 1)

    # FIXME: though limited by extent of chunks and reduced by super-grid chunk
    # size, the index will still be sparse.  Perhaps we would be better to use a
    # point-sparse tensor instead of dense.
    #
    # N-d with shape of shape.
    cindex = torch.zeros(shape, dtype=torch.int32, device=chunk.location.device) - 1
    # Batches of indices into cindex. shape after transpose is (N, Nbatch).
    indices = (super_location - minp).T
    # Count over the batch dimension, shape is (Nbatch,)
    iota = torch.arange(chunk.nbatches, dtype=torch.int32, device=chunk.location.device)
    cindex[*indices] = iota
    # Block assumes batches, so must make one even though it will only be one.
    return Block(location = minp[None,:], data = cindex[None,:])

