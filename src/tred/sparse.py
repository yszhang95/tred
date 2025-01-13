#!/usr/bin/env python
'''
Support for block-sparse binned (BSB) data.


Most functions take singular or batched tensor input and will return tensor
output shaped equivalently.

See docs/concepts.org for a description of the concepts and their terms.

'''


import torch

from .indexing import crop_batched
from . import chunking
from .util import to_tensor, to_tuple
from .blocking import Block
from .types import IntTensor, Shape, Tensor

from collections import defaultdict

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


    def spoint(self, gpoint:IntTensor) -> IntTensor:
        '''
        Return the bin-grid index vector that would contain the
        grid-point index vector in the bin's half-open extent.

        :param gpoint: point-grid index vector.
        :return: bin-grid index vector.

        If input is a 2D tensor, the first dimension runs over batches and the
        output is also a 2D batched tensor.

        Inverse of `gpoint`.
        '''

        return self.to_tensor(gpoint) // self.spacing

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
        

def block_chunk(sgrid: SGrid, block: Block) -> Block:
    '''
    Break up block into aligned chunks.
    '''
    env = sgrid.envelope(block)
    fill_envelope(env, block)
    assert env.data is not None
    return reshape_envelope(env, sgrid.spacing)


class BSB:
    '''
    An N-dimensional tensor-like object with "block sparse bin" storage.

    A BSB implements a super-grid of given spacing.  It then stores user data in
    chunks and provides a subset of tensor operations.
    '''

    spacing: IntTensor
    """1D N-vector of bin shape"""


    def __init__(self, spacing: IntTensor, block: torch.Tensor|None = None):
        '''
        Create a BSB sparse tensor-like object with given super-grid spacing.
        '''
        self.sgrid = SGrid(spacing)
        if block is not None:
            self.fill(block)

            
    def fill(self, block: Block):
        '''
        Fill self with block.

        A `block` may be N-dimensional or 1+N dimensional if batched.  N must be
        same as the size of the spacing vector.
        '''
        if block.vdim != self.sgrid.vdim:
            raise ValueError(f'BSB.fill: dimension mismatch: {block.vdim=} != {self.sgrid.vdim}')

        chunks = block_chunk(self.sgrid, block)
        self.fill_chunks(chunks)


    # fixme: move to chunking.py
    def fill_chunks(self, chunks: Block):
        '''
        Fill self from chunks.
        '''

        # The bins in super-grid points.
        super_locations = self.sgrid.spoint(chunks.location)

        # Find the super-grid bounds that contain all chunks.
        min_point = super_locations.min(dim=0).values
        max_point = super_locations.max(dim=0).values
        shape = to_tuple(max_point - min_point + 1)

        # The map from an super-grid index to an index of a list of chunks.
        # Explicitly on CPU because we use it to index into Python list.
        self.chunks_index = torch.zeros(shape, dtype=torch.int32, device='cpu')
        self.chunks_origin = min_point # super-grid loc of [0]-index

        # intermediate collection of chunks by their super_offset
        by_bin = defaultdict(list)
        for super_loc, data in zip(super_locations, chunks.data):
            cindex = super_loc - self.chunks_origin
            cindex = to_tuple(cindex)
            by_bin[cindex].append(data)
            
        self.chunks = list()
        for cindex, cdatas in by_bin.items():
            cdata = cdatas.pop()
            if len(cdatas):
                cdata = sum(cdatas, cdata)
            if not torch.any(cdata):
                continue
            cind = len(self.chunks) # linear index
            self.chunks.append(Block(data=cdata, location=cindex))
            self.chunks_index[cindex] = cind
            

