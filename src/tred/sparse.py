#!/usr/bin/env python
'''
Support for block-sparse binned (BSB) data.

The following terms used here.

A "point" is an integer-valued N-vector that is interpreted as an absolute
location in an infinite, discrete N-dimensional space.

A "grid" (or "point-grid") is the collection of all possible points.

An "offset vector" or "offset" represents the vector difference from one given
point to another.

An "origin point" or "origin" is a particular point of reference.  An "absolute
origin" is defined as the "zero point" vector with all elements equal to zero.
All other origins are at some offset from the absolute origin.

An "index vector" or "index" is an offset that is constrained to have
non-negative values with upper bounds.  It may be used to index a tensor or to
describe a vector-distance.

A "bounds" is a rectangular subset of a grid identified by two points.  The
lowest corner point represents the location of the bounds.  The offset from
lowest corner to highest corner gives the shape or extent of the bounds.  The
highest corner is one grid spacing higher than the points assumed to be in
bounds.

A "block" is a tensor providing values associated with points in a bounds.

A "super-grid" is derived from a grid.  On its own, a super-grid is a grid but
to keep terms clear the prefix "super-" is given to all grid concepts when they
apply to the super-grid.

The super-origin is identified with the origin (both are zero-points).  

The remaining derivation of the super-grid from the grid is specified by an
index vector defined on the grid called the "spacing".  In general, the point
identified with a given super-point is defined as the element-wise product of
the super-point and the spacing.  Conversely, the super-point associated with
any point is the element-wise integer-division of the point by the spacing.

We call this association a "bin".  A bin may be located in the super-grid by its
super-point or in the grid by the corresponding point.  A bin has half-open unit
extent in the super-grid and in the grid it has half-open extent given by the
spacing.

A *chunk* is a grid block with bounds spanning a single bin of grid points or
equivalently a single super-grid point.

An *envelope* represents a bounds spanning an integer number of chunk bounds.

The code uses a shorthand hint to keep super-grid and grid distinct.  The prefix
"s" is used to indicate a super-grid quantity and "g" to indicate a grid
quantity.  For example a bin may be located by its (absolute) "spoint" or its
"gpoint".  Given a sorigin and a gorigin one may convert between a sindex and a
gindex.

Most functions take singular or batched tensor input and will return tensor
output shaped equivalently.
'''


import torch
from torch import IntTensor, Tensor

from .indexing import crop_batched, chunk, chunk_location
from .util import to_tensor, to_tuple

from collections import defaultdict
import dataclasses


# The torch dtype to use for representing indices.
index_dtype = torch.int32

def bounds(points: IntTensor) -> IntTensor:
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
    return bounds( torch.vstack(points) )


class Block:
    '''
    A block represents scalar values at over a bounds located on a.
    '''
    def __init__(self, data, location):
        self.data = to_tensor(data)
        self.location = to_tensor(location)


class BSB:
    '''
    An N-dimensional tensor-like object with "block sparse bin" storage.

    A BSB implements a super-grid of given spacing.  It then stores user data in
    chunks and provides a subset of tensor operations.
    '''

    spacing: IntTensor
    """1D N-vector of bin shape"""


    def __init__(self, spacing: IntTensor):
        '''
        Create a BSB sparse tensor-like object with given super-grid spacing.
        '''
        self.spacing = to_tensor(spacing)

    def to_tensor(self, thing):
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
        

        
    def batchify(self, block: Block) -> Block:
        '''
        Return block which is assured to be batched.
        '''
        nl = len(block.location.shape)
        nd = len(block.data.shape)

        if nl == 1:             # nominally unbatched
            vdim = block.location.shape[0]
            if vdim != nd:
                raise ValueError(f'unbatched block dimension mismatch {nd} != {vdim}')
            return Block(torch.unsqueeze(block.data,0), torch.unsqueeze(block.location,0))

        bd = block.data.shape[0]
        bl = block.location.shape[0]
        if bd != bl:
            raise ValueError(f'batch size mismatch {bd} != {bl}')


        vdim = block.location.shape[1]
        if vdim+1 != nd:
            raise ValueError(f'batched block dimension mismatch {nd} != {vdim+1}')

        return block



    def make_envelope(self, block: Block) -> Block:
        '''
        Return new block that has been padded to align to spacing
        '''
        block = self.batchify(block)
        print(f'{block.data.shape=} {block.location=}')
        nbatches = block.data.shape[0]
        inshape = to_tensor(block.data.shape[1:])
        print(f'{inshape=}')

        # Find the minimum bin span that contains the block.
        # Find the location of the bin containing the lowest point.
        minpts = self.gpoint(self.spoint(block.location)) 
        # Find the location of the bin one past the bin containing the highest point.
        maxpts = self.gpoint(self.spoint(block.location+inshape-1)+1)
        shapes = maxpts - minpts
        print(f'{minpts=} {shapes=}')

        # Calculate the bin-aligned bounds large enough to hold all batches.
        env_shape = to_tuple(torch.max(shapes, dim=0).values)
        env_shape_batched = tuple([nbatches] + list(env_shape))

        env_data = torch.zeros(env_shape_batched, dtype=block.data.dtype, device=self.spacing.device)
        env_loc = minpts
        print(f'{env_shape_batched=}')

        # get the location of the block in its envelope
        env_offset = block.location - env_loc
        print(f'{env_shape=} {env_loc=} {env_offset=}')

        # Set block values in the envelope
        inds = crop_batched(env_offset, inshape, env_shape)
        print(f'cropped inds {inds.shape=}')
        print(f'{env_data.shape=} {block.data.shape=}')
        env_data.flatten()[inds] = block.data.flatten()
        
        return Block(env_data, env_loc)


    def fill(self, block: Block):
        '''
        Fill self with arbitrary blocks.  
        '''
        block = self.batchify(block)
        env = self.make_envelope(block)
        chunks = self.make_chunks(env)
        self.fill_chunks(chunks)
        

    def make_chunks(self, env: Block) -> Block:
        '''
        Construct chunks from the envelope.
        '''
        env = self.batchify(env)

        env_shape = env.data.shape[1:]
        chunk_shape = self.spacing
        vdim = len(chunk_shape)
        return Block(chunk(env.data, chunk_shape).flatten(0, vdim),
                     chunk_location(env.location, env_shape, chunk_shape).flatten(0, vdim))


    def fill_chunks(self, chunks: Block):
        '''
        Fill self from chunks.
        '''

        chunks = self.batchify(chunks)

        # The bins in super-grid points.
        super_locations = self.spoint(chunks.location)

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
            self.chunks.append(Block(cdata, cindex))
            self.chunk_index[cindex] = cind
            


    def _fill_monolithic(self, block: Block) -> Block:
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
        # - convert envelop locations to bin locations (chunks).
        # - collect all chunks at a location.
        # - sum chunks at each location
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
        minpts = self.gpoint(self.spoint(location)) 
        # Find the location of the bin one past the bin containing the highest point.
        maxpts = self.gpoint(self.spoint(location+inshape-1)+1)
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
        envelope.flatten()[inds] = block.flatten()


        # Break the large envelope into smaller chunks (per-bin tensors)
        chunk_shape = self.spacing
        vdim = len(chunk_shape)
        chunks = chunk(envelope, chunk_shape).flatten(0, vdim)
        clocations = chunk_location(env_loc, env_shape, chunk_shape).flatten(0, vdim)

        # These hold indices of the bin grid (not the point grid)
        chunk_min_point = torch.round(minpts.min(dim=0).values / chunk_shape).to(dtype=torch.int32)
        chunk_max_point = torch.round(maxpts.max(dim=0).values / chunk_shape).to(dtype=torch.int32)
        chunk_space_shape = tuple((chunk_max_point - chunk_min_point).tolist())
        # keep on CPU this is meant for indexing the chunk lists.
        chunk_space = torch.zeros(chunk_space_shape, dtype=torch.int32, device='cpu')

        # Collect chunks by their bin.
        by_bin = defaultdict(list)
        # need tuples for the keys so bring it all home in a big transfer
        chunk_locations = torch.round(clocations/chunk_shape).to(dtype=torch.int32, device='cpu') 
        for cloc, cten in zip(chunk_locations, chunks):
            cloc = tuple(cloc.tolist())
            by_bin[cloc].append(cten)
        
        chunk_list = list()
        chunk_locs = list()
        for cindex, (cloc, to_add) in enumerate(by_bin.items()):
            ten = to_add.pop(0)
            if len(to_add):
               ten = sum(to_add, ten)
            if not torch.any(ten):
                continue

            chunk_list.append(ten)
            chunk_locs.append(cloc)
            clocind = tuple((torch.tensor(cloc) - chunk_min_point).tolist())
            print(f'{clocind=}')
            chunk_space[clocind] = cindex

        self.chunk_space = chunk_space
        self.chunk_origin = chunk_min_point
        self.chunk_list = chunk_list
        self.chunk_locs = chunk_locs

