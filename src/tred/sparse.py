#!/usr/bin/env python
'''
Support for sparse data.
'''
from typing import Tuple, List

import torch
from torch import IntTensor

from .indexing import crop_batched


index_dtype = torch.int32
#: The torch dtype to use for representing indices.


class BSB:
    '''
    An N-dimensional tensor-like object with "block sparse bin" storage.

    The storage model is optimized for the case where nonzero elements are
    locally dense while globally sparse.  The parameter "bsize" ("bin size" or
    "bin shape" as it is N-dimensional) defines the boundary between these two
    scales.

    The two scales are connected by a grid aka the "point-grid" and a coarser
    "super-grid" aka "bin-grid".  The bin-grid has spacing given by the integer
    N-vector "bsize".  Locations in both grids are represented by an "index
    vector" which is a non-negative integer N-vector of grid point indices.

    The point-grid index vector is sometimes called a "pindex" and the blob-grid
    index vector a "bindex".  The two grids are aligned so that the zero pindex
    and zero bindex are at the same grid point.

    To emphasize, a bin is located on the point-grid at the pindex=bindex*bsize
    of its lower corner and vice versa the bin is located in the bin-grid as
    determined by its lower corner as bindex=pindex/bsize.  A bin covers a
    half-open point-grid range (volume) from pindex to pindex+binsize.

    Conceptually, the BSB is semi-infinite and its possible non-negative indices
    may grow arbitrarily.

    We will say "vdim" to identify the number of dimensions of the grids (eg, 1,
    2 or 3).  Other dimension counts refer to tensor dimensions.  A "vector" is
    a 1D tensor of length vdim.  An "index" is an integer vector locating an
    element of the "point-grid" or the "bin-grid".  Method arguments are
    expressed in the singular but can usually take tensors with an additional
    batched dimension.
    '''

    bsize: IntTensor
    """1D N-vector of bin shape"""


    def __init__(self, bsize: IntTensor):
        '''
        Create a BSB sparse tensor-like object with given bin size.

        Args:

        :param bsize: 1D N-vector giving bin size in units of number of grid points per dimension.

        '''
        if not isinstance(bsize, torch.Tensor):
            bsize = torch.tensor(bsize)

        self.bsize = bsize.to(dtype=index_dtype)


        #: When data is added self will add .extent, .blocks, .offsets, .bins.

    def bindex(self, pindex:IntTensor) -> IntTensor:
        '''
        Return the bin-grid index vector that would contain the
        grid-point index vector in the bin's half-open extent.

        :param pindex: point-grid index vector.
        :return: bin-grid index vector.

        If input is a 2D tensor, the first dimension runs over batches and the
        output is also a 2D batched tensor.

        Inverse of `pindex`.
        '''

        return pindex // self.bsize

    def pindex(self, bindex: IntTensor) -> IntTensor:
        '''
        Return the bin's grid index (bin's lower corner).

        :param bindex: bin-grid index vector.
        :return: point-grid index vector.

        If input is a 2D tensor, the first dimension runs over batches and the
        output is also a 2D batched tensor.

        Inverse of `bindex`.
        '''
        return bindex * self.bsize
        
    def pbounds(self, points: IntTensor) -> IntTensor:
        '''
        Return a grid point box that bounds the points.

        :param points: 2D tensor of point-grid index N-vectors
        :return: a 2D tensor of shape (2,N) of two point-grid index N-vectors.

        This returns a min/max pair of vectors.  The max is 1 grid point larger
        than the index coordinates of all points.

        The input is may be a single point given as a 1D tensor.
        '''
        if isinstance(points, torch.Tensor):
            if len(points.shape) == 1: # singular
                return points, points+1
            # batched
            return torch.vstack((torch.min(points, dim=0).values, torch.max(points, dim=0).values + 1))
        # assume collection of N-d points
        return self.pbounds( torch.vstack(points) )

    def fill(self, block: IntTensor, location: IntTensor) -> Tuple[IntTensor, IntTensor]:
        '''
        Store a block of data at given location.

        :param block: values defined on the point-grid.
        :param location: point-grid location of block.

        The inputs may be singular or have a batched dimension.  A singular
        block is an N-dimensional tensor and a singular location is a 1D tensor
        of an N-vector point-grid index vector.  If batched, the batch dimension
        is added to both as dimension-0.
        '''
        
        # The algorithm works over these stages:
        # - find envelope that is aligned to bin-grid and large enough to hold all blocks.
        # - calculate index locations in envelope for blocks
        # - copy blocks into envelope
        # - reshape envelope to bin dimensions
        # - convert envelop locations to bin locations.
        # - collect overlapping bins by their location
        # - sum overlaps
        # - store and return
        #
        # fixme: factor from this monolith into multiple methods....

        # assure inputs are batched 
        if len(location.shape) == 1:
            location = location[None,:]
            block = block[None,:]

        nbatches = block.shape[0]
        inshape = torch.tensor(block.shape[1:])

        # Find the minimum bin span that contains the block.
        # Find the location of the bin containing the lowest point.
        minpts = self.pindex(self.bindex(location)) 
        # Find the location of the bin one past the bin containing the highest point.
        maxpts = self.pindex(self.bindex(location+inshape-1)+1)
        shapes = maxpts - minpts

        # Calculate an envelope of bin-aligned spans.  The envelope is simply a
        # bin-aligned span large enough to hold the per block bin-aligned spans.
        env_shape = tuple(torch.max(shapes, dim=0).values.tolist())
        env_shape_batched = tuple([nbatches] + list(env_shape))
        envelope = torch.zeros(env_shape_batched, dtype=block.dtype)

        # location of each block's envelope
        env_loc = minpts
        
        # The location of the box in the envelope holding the block
        env_ind0 = location - env_loc

        # To set values inside the envelope:
        # - flatten the envelope and block
        # - calculate envelope indices corresponding to the flattened block
        inds = crop_batched(env_ind0, inshape, env_shape)
        envelope.flatten()[inds] = block.flatten();


        print("chunk is not tested")
        return envelope, env_loc

