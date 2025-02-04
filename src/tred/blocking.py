#!/usr/bin/env python
'''
tred.blocking provides functions for N-dimensional "blocks".  See `Block`.
'''

from tred.util import to_tensor
import torch
from .types import IntTensor, Tensor, Shape, Size, index_dtype

class Block:
    '''
    A batch of rectangular, N-dimensional volumes at locations in a discrete
    integer space (grid).

    Volume are restricted to share a common (unbatched) `shape`.
    '''
    def __init__(self, location:IntTensor, shape: Shape|None=None, data:Tensor|None=None):
        '''
        Create a Block.

        The `location` of the volumes may be 1D (unbatched) or 2D (batched)
        N-vector.  If unbatched, a single-batch version is stored.  If not
        proved as a tensor, it is coerced to an IntTensor.

        The `shape` gives the common volume shape.

        The `data` provides a tensor of values defined on the volumes.  It will
        be stored in batched form consistent with that of `location`.

        The `shape` argument is only considered if `data` is not provided.
        '''
        if not isinstance(location, Tensor):
            location = torch.tensor(location, dtype=index_dtype)

        if len(location.shape) == 1:
            location = location.unsqueeze(0)
        if len(location.shape) != 2:
            raise ValueError(f'Block: unsupported shape for volume locations: {location.shape}')
        self.location = location

        if data is None:
            self.set_shape(shape)
        else:
            self.set_data(data)

    def __str__(self):
        return f'<block {self.nbatches} of shape {self.shape}>'

    def size(self):
        '''
        Return torch.Size like a tensor.size() does. This includes batched dimension.
        '''
        return Size([self.nbatches] + self.shape.tolist())

    @property
    def vdim(self):
        '''
        Return the number of spacial/vector dimensions, excluding batch dimension
        '''
        return self.location.shape[1]

    @property
    def nbatches(self):
        '''
        Return the number of batches.
        '''
        return self.location.shape[0]

    @property
    def device(self):
        '''
        Return device of location.
        '''
        return self.location.device

    def set_shape(self, shape:Shape):
        '''
        Set the spacial shape of the volumes.  This will drop any data.
        '''
        if hasattr(self, "data"):
            delattr(self, "data")

        if not isinstance(shape, Tensor):
            shape = self.to_tensor(shape, dtype=index_dtype)
        else:
            shape = shape.to(device=self.location.device, dtype=index_dtype)

        if len(shape.shape) != 1:
            raise ValueError(f'Block: volume shape must not be batched: {shape.shape}')
        vdim = self.vdim
        if len(shape) != vdim:
            raise ValueError(f'Block: volume shape has wrong dimensions: {len(shape)} expected {vdim}')
        self.shape = shape
        
    def set_data(self, data:Tensor):
        '''
        Set data for block.
        '''
        if not isinstance(data, Tensor):
            raise TypeError('Block: data must be a tensor')

        vdim = self.vdim

        if len(data.shape) == vdim:
            data = data.unsqueeze(0)
        if len(data.shape) != vdim+1:
            raise ValueError(f'Block: illegal data shape: {data.shape} for volume {vdim}-dimension volumes')
        nbatch = self.nbatches
        if data.shape[0] != nbatch:
            raise ValueError(f'Block: batch size mismatch: got {data.shape[0]} want {nbatch}')

        self.set_shape(data.shape[1:])
        self.data = data

    def to_tensor(self, thing, dtype=index_dtype):
        '''
        Return thing as tensor on same device as location tensor.
        '''
        return to_tensor(thing, device=self.location.device, dtype=dtype)


def apply_slice(block: Block, space_slice) -> Block:
    '''
    Apply a slice to the block data along the spatial dimensions.

    The space_slice is an N-tuple with a slice instance for each spatial
    dimensions.  The slice is common to all elements along the batch dimension.
    The returned Block.location records the input plus the offset defined by the
    "starts" of the slices.
    '''
    offset = block.to_tensor([s.start or 0 for s in space_slice])
    return Block(location = block.location + offset,
                 data = block.data[:,*space_slice])

def batchify(ten: Tensor, ndim: int) -> tuple:
    '''
    Return tuple of (tensor,bool).

    The tensor is returned in batch form with ndim+1 dimensions.  True is
    returned only if the batch dimension was added.  An added batch dimension is
    dimension 0.
    '''
    if len(ten.shape) == ndim:
        return ten.unsqueeze(0), True
    return ten, False

