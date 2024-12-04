#!/usr/bin/env python
'''
Support for sparse data.
'''

class BSB:
    '''
    A N-dimensional tensor-like object with block sparse bin storage.
    '''

    index_dtype = torch.int32

    def __init__(self, binsize, offset=None):
        '''
        binsize gives shape of the bins in units of grid points.
        offset gives a gridpoint to be considered as an origin for the bins.
        '''
        self.binsize = binsize.to(dtype=self.index_dtype)
        self.ndim = len(binsize)
        self.offset = offset or torch.tensor([0]*self.ndim, dtype=self.index_dtype)

    def bindex(self, gridpoint):
        '''
        Return the bin index that would contain the grid point.
        '''
        return (gridpoint - self.offset) // self.binsize

    def pindex(self, binpoint):
        '''
        Return the point at the low end of the bin.
        '''
        return binpoint * self.binsize + self.offset
        
    def bounds(self, points):
        '''
        Return a box that bounds the points.  
        '''
        if isinstance(points, torch.Tensor):
            ndim = len(points.shape)
            if ndim == self.ndim:  # single point
                return points, points+1
            if ndim == self.ndim + 1:  # batched
                return torch.min(self.points, dim=0), torch.max(self.points, dim=0) + 1
            raise ValueError(f'illegal shape {points.shape} for {self.ndim} dimensions')
        # assume collection of N-d points
        return self.bounds( torch.vstack(points) )

    def reserve(self, box):
        '''
        Ensure storage spans the given grid point box
        '''
        if not hasattr(self, "bins"):
            # green field
            self.bounds = (self.bindex(lo), self.bindex(hi-1)+1)


            self.extend = extent
            self.blocks = list()
            self.offsets = list()
            self.bins = torch.tensor(

        lo, hi = extent
        if 
        
