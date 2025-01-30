#!/usr/bin/env python

# fixme: make these functions return blocking.Block

'''
Methods to raster point-like electron distributions (depos) to an N-dimensional grid.

Each function is called similarly:

  object_method(grid, objects, **kwds) -> [(output, offset)]

  - grid :: real 1D tensor N-vector providing the grid spacing of a regular,
    rectangular N-dimensional grid.  Origin of grid (grid indices [0]*N) is
    assumed to be at the origin of space ie,

  - objects :: real 2D tensor of shape (N_params, N_objects) describing objects
    of a certain type.

  - kwds :: dict of optional keyword parameters specific to the particular
    method.

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


def binned_1d(grid, centers, widths, q, nsigma=None, minbins=None):
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
    - nsigma :: minimum multiple of Gaussian sigma to cover.
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


def binned_nd(grid, centers, sigmas, charges, nsigma=None, minbins=None):
    '''
    N-dimensional (N>1)

    - grid :: real scalar or 1D N-vector of grid spacing
    - centers :: 1D (N_depos,) or 2D (N_depos, N) tensor of centers 
    - sigmas :: 1D (N_depos), or 2D (N_depos, N) tensor of Gaussian sigma widths per depo (can have zeros)
    - charges :: 1D tensor of total depo charge (N_depos,)
    - nsigma :: scalar or 1D (N,) minimum multiple of Gaussian sigma to cover.
    - minbins :: 1D N-vector of per-dimension absolute minimum number of grid points to cover half the Gaussian.

    Return tuple (raster, offset) 

    '''
    if not isinstance(grid, torch.Tensor):
        grid = torch.tensor([grid]) # make 1D (vdim,)
    vdims = len(grid)
    if len(centers.shape) == 1:
        centers = centers[:,None] # add 1D vector dimension
    if len(sigmas.shape) == 1:
        sigmas = sigmas[:,None]
    if nsigma is None:
        nsigma = 3.0
    if not isinstance(nsigma, torch.Tensor):
        nsigma = torch.tensor([nsigma]*vdims)

    if centers.shape[1] != vdims:
        raise ValueError(f'wrong shape: {centers.shape=} for {vdims} vector dims')
    if sigmas.shape[1] != vdims:
        raise ValueError(f'wrong shape: {sigmas.shape=} for {vdims} vector dims')


    # Find number of grid points that span half the largest Gaussian.
    # (ndepos, vdims)
    n_half = torch.round(sigmas*(nsigma/grid))

    if minbins is not None:
        # (ndepos+1, vdims)
        n_half = torch.vstack((n_half, minbins))
    # (vdims, )
    n_half = torch.max(n_half.to(dtype=torch.int32), dim=0).values

    # Grid index nearest each center.
    # (ndepos, vdims)
    gridc = torch.round(centers/grid).to(dtype=torch.int32)

    # Grid index at the starting point (lowest corner grid point) for each region.
    grid0 = gridc - n_half

    # Location of grid point nearest center relative from start corner point.
    rel_gridc = gridc - grid0

    # Suffer per-dimension serialization.
    # We do it because linspace() is 1D only.
    # Perhaps can remove loop if refactor to use meshgrid. 
    integs=list()
    for dim in range(vdims):

        dim_n_half = n_half[dim]

        # Enumerate the grid points covering the two halves of this dim's Gaussian
        # Note, add 1 as we do a shift-by-one subtraction after erf()'s below
        # (npts,)
        rel_grid_ind = torch.linspace(0, 2*dim_n_half, 2*dim_n_half+1)

        abs_grid_pts = ((grid0[:,dim][:,None] + rel_grid_ind[None,:]) - 0.5) * grid[dim]
        spikes = sigmas[:,dim] == 0 # depos with zero width

        normals = (abs_grid_pts - centers[:,dim][:,None])/sigmas[:,dim][:,None]
        normals[spikes] = 0     # remove inf
        erfs = torch.erf(normals)
        integ = 0.5*(erfs[:, 1:] - erfs[:, :-1])
        integ[spikes] = 0
        integ[spikes, rel_gridc[spikes, dim]] = 1.0
        integs.append(integ)

    # integs = [torch.unsqueeze(one, dim=0) for one in integs]
    if vdims == 1:
        integ = integs[0]
        charges = charges.reshape(-1, 1)
    elif vdims == 2:
        integ = torch.einsum('ij,ik -> ijk', *integs)
        charges = charges.reshape(-1, 1, 1)
    elif vdims == 3:
        integ = torch.einsum('ij,ik,il -> ijkl', *integs)
        charges = charges.reshape(-1, 1, 1, 1)
    else:
        raise ValueError(f'unsupported vector dimensions: {vdim}')

    qeff = charges*integ
    return (qeff, grid0)



def binned(grid, centers, sigmas, charges, nsigma=None, minbins=None):
    '''
    Raster depos by integrating Gaussian distribution around each grid point 

    nsigma is either scalar or N-dimensional giving the number of sigma over
    which to perform the integration.
    '''
    # ngrid = len(grid)
    # if ngrid == 1:              # special case for now
    #     return binned_1d(grid, centers, sigmas, charges, nsigma, minbins)
    return binned_nd(grid, centers, sigmas, charges, nsigma, minbins)

