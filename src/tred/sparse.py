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

