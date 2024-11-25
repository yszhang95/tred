#!/usr/bin/env python
'''
Methods to raster electron distributions to an N-dimensional grid.

Each function is called similarly:

  object_method(grid, objects, **kwds) -> [(output, offset)]

  - grid :: real 1D tensor N-vector providing the grid spacing of a regular,
    rectangular N-dimensional grid.  Origin of grid (grid indices [0]*N) is
    assumed to be at the origin of space ie,

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

The offset is a vector of integer indices assuming 0 is at the origin for a
given dimension.  It may have negative values.  Multiply by grid to get absolute
location of grid point.

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


def depo_binned_1d(grid, centers, widths, q, nsigma=None, minbins=None):
    '''
    1D.

    Integrate Gaussian depos near grid points.

    Raster consists of total charge in the +/- 1/2 grid spacing around each grid
    point for points within +/- nsigma*width of center.

    Offset gives the grid index of the first grid point.  Ie, the grid point
    near the location given by center-nsigma*width.

    Note, bins are centered on grid points.

    - grid :: 1D N-vector of grid spacing
    - centers :: 1D tensor of centers (N_depos,)
    - widths :: 1D tensor of Gaussian sigma widths (N_depos,)  (can be zero)
    - q :: 1D tensor of total depo charge (N_depos,)
    - nsigma :: per-depo minimum multiple of Gaussian sigma to cover.
    - minbins :: per-dimension absolute minimum number of grid points to cover half the Gaussian.
    '''
    if nsigma is None:
        nsigma = 3.0

    # Find number of grid points that span half the largest Gaussian.
    n_half = torch.round((widths*nsigma)/grid).to(dtype=torch.int32)
    if minbins is not None:
        n_half = torch.vstack((n_half, minbins))
    n_half = torch.max(n_half)

    # find the grid index nearest to each center.
    gridc = torch.round(centers/grid).to(dtype=torch.int32)

    # find the grid index at the starting point for each region.
    grid0 = gridc - n_half

    rel_gridc = gridc - grid0

    # Enumerate the grid points covering the two halves.
    # Note, add 1 as we do a shift-by-one subtraction below.
    rel_grid_ind = torch.linspace(0, 2*n_half, 2*n_half + 1)
    rel_grid_ind = torch.unsqueeze(rel_grid_ind, 0)

    # (ndepos, 1=vdim)
    grid0 = torch.unsqueeze(grid0, -1)

    # (ndepos, npoints)
    # Points at which to sample the Gaussians.
    abs_grid_pts = ((grid0 + rel_grid_ind) - 0.5) * grid

    centers = torch.unsqueeze(centers, -1)
    spikes = widths==0
    widths = torch.unsqueeze(widths, -1)

    # Transform grid points on Gaussian to equivalent points on a Normal distribution.
    normals = (abs_grid_pts - centers)/widths
    normals[spikes] = 0

    erfs = torch.erf(normals)
    integ = 0.5*(erfs[:, 1:] - erfs[:, :-1])
    integ[spikes] = 0
    integ[spikes, rel_gridc[spikes]] = 1.0
    raster = torch.unsqueeze(q, -1)*integ
    return (raster, torch.squeeze(grid0))


def depo_binned_nd(grid, centers, widths, q, nsigma=None,
                   minbins=None):
    '''
    N-dimensional (N>1)

    - grid :: 1D N-vector of grid spacing
    - centers :: 2D tensor of centers (N_depos, N)
    - widths :: 2D tensor of Gaussian sigma widths per depo (N_depos, N)  (can have zeros)
    - q :: 1D tensor of total depo charge (N_depos,)
    - nsigma :: per-depo minimum multiple of Gaussian sigma per dimension to cover.
    - minbins :: 1D N-vector of per-dimension absolute minimum number of grid points to cover half the Gaussian.

    '''
    ndims = len(grid)
    if nsigma is None:
        nsigma = torch.tensor([3.0]*ndims)

    # Find number of grid points that span half the largest Gaussian.
    n_half = torch.round((widths*nsigma)/grid)
    if minbins is not None:
        n_half = torch.vstack((n_half, minbins))
    n_half = torch.max(n_half.to(dtype=torch.int32), dim=0).values

    # Grid index nearest each center.
    gridc = torch.round(centers/grid).to(dtype=torch.int32)
    print(f'{gridc.shape=}\n{gridc}')    

    # Grid index at the starting point (lowest corner grid point) for each region.
    grid0 = gridc - n_half
    print(f'{grid0.shape=}\n{grid0}')

    # Location of grid point nearest center relative from start corner point.
    rel_gridc = gridc - grid0
    print(f'{rel_gridc.shape=}\n{rel_gridc}')

    # Suffer per-dimension serialization.
    # This perhaps could be parallelized if refactored to use meshgrid. 
    integs=list()
    for dim in range(ndims):

        # Enumerate the grid points covering the two halves of this dim's Gaussian
        # Note, add 1 as we do a shift-by-one subtraction after erf()'s below
        rel_grid_ind = torch.linspace(0, 2*n_half[dim], 2*n_half[dim]+1)
        # Add the per-depo axis
        rel_grid_ind = torch.unsqueeze(rel_grid_ind, 0)
        print(f'{dim=} {rel_grid_ind.shape=}\n{rel_grid_ind}')
                                      
        dim_grid0 = torch.unsqueeze(grid0[:,dim], -1)
        print(f'{dim_grid0.shape=} {dim_grid0}')

        abs_grid_pts = ((dim_grid0 + rel_grid_ind) - 0.5) * grid[dim]
        print(f'{abs_grid_pts.shape=}\n{abs_grid_pts}')

        dim_centers = torch.unsqueeze(centers[:,dim], -1)
        spikes = widths[:,dim] == 0
        dim_widths = torch.unsqueeze(widths[:,dim], -1)

        normals = (abs_grid_pts - dim_centers)/dim_widths
        normals[spikes] = 0
        erfs = torch.erf(normals)
        integ = 0.5*(erfs[:, 1:] - erfs[:, :-1])
        integ[spikes] = 0
        integ[spikes, rel_gridc[spikes, dim]] = 1.0
        integs.append(integ)

    # integs = [torch.unsqueeze(one, dim=0) for one in integs]

    if ndims == 2:
        ein = 'ij,ik -> ijk'
        q = q.reshape(-1, 1, 1)
    else:
        eign = 'ij,ik,il -> ijkl'
        q = q.reshape(-1, 1, 1, 1)
    integ = torch.einsum(ein, *integs)

    print(f'{q.shape=} {integ.shape=}')

    # return (q.reshape(-1,1,1)*integ, grid0)
    return (q*integ, grid0)



def depo_binned(grid, c, w, q, nsigma=None, minbins=None):
    '''
    Raster depos by integrating Gaussian distribution around each grid point 

    nsigma is either scalar or N-dimensional giving the number of sigma over
    which to perform the integration.
    '''
    ngrid = len(grid)
    if ngrid == 1:              # special case for now
        return depo_binned_1d(grid, c, w, q, nsigma, minbins)

    return __dir__[f'depo_binned_{ngrid}d'](grid, c, w, q, nsigma)
