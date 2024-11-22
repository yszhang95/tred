#!/usr/bin/env python
'''
Methods to raster electron distributions to an N-dimensional grid.

Each function is called similarly:

  object_method(grid, objects, **kwds) -> [(output, offset)]

  - grid :: real 1D N-tensor providing the spacing of a regular rectangular
    N-dimensional grid. Origin of grid is [0]*N

  - objects :: real 2D tensor of shape (N_params, N_objects) describing objects
    of a certain type.

  - kwds :: dict of optional keyword parameters specific to the particular
    method.

Function naming convention <object>_<method> with <object> naming the type of
input object eg "depo" or "step" and <method> naming the raster algorithm.

These functions take N-dimensional grids based on grid tensor size may delegate
to a specific N-dimensional function <object>_<method>_<N>d().  Not all
combinations are supported.

Functions return a value that is a list of tuples.  Each tuple gives (output,offset)

- output :: an N-D tensor of effective number of electrons on grid points
- offset :: a 1D N-tensor giving offset from grid origin to lower corner of output

Notes for caller:

The units of the "grid" spacing must correspond to any location and size
parameters represented in the objects tensor.  For example, if the simulation
assumes a drift along the X-axis then object X-location and X-width may be in
units of time while spacial units are used for parameters relevant to the
transverse dimensions.  The methods do not depend on the units assumed by the
caller.

May change this:

A depo objects tensor has the following 2N+1 rows
- (c1,c2,...,cN) :: N coordinates of the center of a Gaussian distribution.
- (w1,w2,...,wN) :: N Gaussian sigma widths of distribution along an axis.
- ne :: integral number of electrons associated with the depo

A step object tensor has the following 3N+1 rows
- (b1,b2,...,bN) :: N coordinates of the begin point of the step
- (e1,e2,...,eN) :: N coordinates of the end point of the step
- (w1,w2,...,wN) :: N Gaussian sigma widths of distribution along an axis.
- ne :: integral number of electrons associated with the step
'''

import torch


def depo_binned_1d(grid, centers, widths, q, nsigma=torch.tensor([3]),
                   minbins=torch.tensor([3], dtype=torch.int32)):
    '''
    1D.

    - grid :: 1D 1-tensor of grid spacing
    - c :: 1D tensor of centers (N_depos,)
    - w :: 1D tensor of Gaussian widths (N_depos,)
    - q :: 1D tensor of total depo charge (N_depos,)
    - minbins :: minimum number of bins to cover from center of depo
    '''
    
    nbins_half = (1 + (w*nsigma)/grid).to(dtype=torch.int32)
    if nbins_half[0] < minbins: # make sure at least one entry is bigger than minbins
        nbins_half[0] = int(minbins)
    nbins_half = torch.max(nbins_half)

    # grid index nearest Gaussian center.

    gridc = torch.round(centers/grid).to(dtype=torch.int32)
    grid0 = gridc - nbins_half

    local_grid = torch.linspace(-nbins_half, nbins_half, 2*nbins_half + 1)*grid
    print(f'{local_grid=}')
    local_norm = local_grid / widths.reshape(-1, 1)

    erfs = torch.erf(local_norm)
    integ = 0.5*(erfs[:, 1:] - erfs[:, :-1])
    return (q.reshape(-1,1)*integ, grid0)


def depo_binned_2d(grid, centers, widths, q, nsigma=None,
                   minbins=None, dtype=torch.int32):
    '''
    2D

    grid - 1D 2-tensor of grid spacing
    centers - tensor of 2D centers per depo (N_depos, 2)
    widths - tensor of 2D widths per depo (N_depos, 2)
    q - 1D tensor of charge per depo (N_depos,)
    nsigma - tensor of 2D number of sigma per depo (N_depos, 2)
    minbins - 2D tensor giving minimum number of bins for all depos
    '''
    ndims = len(grid)
    if nsigma is None:
        nsigma = torch.tensor([3.0]*ndims)

    nbins_half = (1 + (widths*nsigma)/grid).to(dtype=torch.int32)
    print(f'{nbins_half=}')
    if minbins is not None:
        nbins_half = torch.vstack((nbins_half, minbins))
    nbins_half = torch.max(nbins_half, dim=0).values

    integs=list()
    for dim in range(len(grid)):
        local_grid = torch.linspace(-nbins_half[dim], nbins_half[dim], 2*nbins_half[dim] + 1)*grid[dim]
        print(f'{local_grid.shape=}')
        print(f'{local_grid=}')
        w = widths[:,dim].reshape(-1,1)
        local_norm = local_grid / w
        print(f'{local_norm.shape=}')
        print(f'{local_norm=}')
        erfs = torch.erf(local_norm)
        integ = 0.5*(erfs[:, 1:] - erfs[:, :-1])
        print(f'{integ.shape=}')
        print(f'{integ=}')
        integs.append(integ)
    print(f'{len(integs)}')
    integ = torch.einsum('ij,ik -> ijk', *integs)
    print(f'{integ.shape=}')
    print(f'{integ=}')

    
    gridc = torch.round(centers/grid).to(dtype=torch.int32)
    print(f'{gridc=}')
    grid0 = gridc - nbins_half
    print(f'{grid0=}')
    print(f'{q.shape} {q}')
    return (q.reshape(-1,1,1)*integ, grid0)


    

def depo_binned_3d(grid, c, w, q, nsigma=3):
    '''
    '''
    raise NotImplemented

def depo_binned(grid, c, w, q, nsigma=3):
    '''
    Raster depos by integrating Gaussian distribution around each grid point 

    nsigma is either scalar or N-dimensional giving the number of sigma over
    which to perform the integration.
    '''
    ngrid = len(grid)
    return __dir__[f'depo_binned_{ngrid}d'](grid, c, w, q, nsigma)
