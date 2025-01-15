#!/usr/bin/env python
'''
A class to hold block-sparse binned data.
'''
from .types import IntTensor
from .sparse import SGrid, block_chunk, index_chunks
from .blocking import Block
from. import chunking 
import torch

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
        self.chunks = chunking.accumulate(chunks)
        self.cindex = index_chunks(self.sgrid, chunks)

