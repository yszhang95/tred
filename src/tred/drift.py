#!/usr/bin/env python
'''
Functions to apply drift related models.

Array shapes are described with:

- npts :: the size of a tensor-dimension spanning a set of points (eg depos/steps).
- vdim :: the size of a tensor-dimension spanning spatial vector-dimensions.

Eg: (npts=5,vdim=3) is 5 points in 3-space.

'''

import torch
from torch.distributions.binomial import Binomial


def transport(locs, target, velocity):
    '''
    Return the time to drift from initial locations (locs) on some axis to a
    target location on that axis at the given (signed) velocity.

    - locs :: real 1D tensor (npts,)
    - target :: real scalar
    - speed :: real scalar
    '''
    return (target - locs)/velocity


def diffuse(dt, diffusion, sigma=None):
    '''
    Return new Gaussian diffusion sigma after drifting by time dt.

    - dt :: real 1D tensor of drift times (npts,).
    - diffusion :: real scalar or 1D (vdim,) tensor of diffusion coefficients.
    - sigma :: real 1D (npts) or 2D (npts,vdim) tensor or None.

    A sigma of None indicate no prior diffusion.

    Note, negative dt entries will result in a shrinking of any initial sigma.
    Where this shrinkage may go negative the resulting sigma element is set to
    zero.
    '''
    squeeze = False
    # eg, diffusion is 5, [5] or [4,5,6]
    if not isinstance(diffusion, torch.Tensor):
        diffusion = torch.tensor([diffusion])
        squeeze = True

    diffusion = diffusion[None,:] # add npts dimension
    dt = dt[:,None]               # add vdim dimension

    if sigma is None:
        sigma = torch.sqrt(2*diffusion*dt)
    else:
        if len(sigma.shape) == 1:
            sigma = sigma[:,None] # add vdim
        sigma = torch.sqrt(2*diffusion*dt + sigma*sigma)
    sigma[torch.isnan(sigma)] = 0
    if squeeze:
        sigma = torch.squeeze(sigma)
    return sigma

def absorb(charge, dt, lifetime, fluctuate=False):
    '''
    Apply an absorption lifetime return counts of surviving electrons (charge).

    - charge :: positive integer 1D tensor (npts,) 
    - dt :: real 1D tensor (npts,) of time each point is subject to absorption.
    - lifetime :: scalar value giving exponential lifetime
    - fluctuate :: if true, apply binomial fluctuation based on mean lost q

    Note, negative dt will lead to exponential increase not decrease.  Ie,
    implies that the drift "backed up" the point.
    '''
    charge = charge.to(dtype=torch.int32)
    if fluctuate:
        loss = 1.0 - torch.exp(-dt / lifetime)
        loss = torch.clamp(loss, 0, 1, out=loss)
        b = Binomial(charge, loss)
        return charge - b.sample()

    return charge * torch.exp(-dt / lifetime)


def drift(locs, velocity, diffusion, lifetime, target=0,
          times=None, vaxis=0, charge=None, sigma=None, fluctuate=False):
    '''
    Apply drift models.

    Returns tuple of post-drift quantities: (drifted, times, sigma, charges)

    Required arguments:

    - locs :: 1D (npts,) or 2D (npts,vdim) tensor of initial locations.
    - velocity :: real scalar (signed) velocity from initial to final location.
    - diffusion :: real scalar or 1D (vdim,) diffusion coefficient.
    - lifetime :: positive real scalar electron absorption lifetime

    Optional arguments:

    - target :: real scalar, location on vaxis to cease drift.  Default is 0.
    - times :: 1D (npts,) tensor of initial times.  Default is zeros.
    - vaxis :: int scalar.  The vector-axis on which to drift.  Default is 0.
    - charge :: real scalar or 1D (npts,) initial charge.  Default is 1000 electrons.
    - sigma :: real 1D (npts,) or 2D (npts,vdim) tensor giving initial sigma.  Default is none.
    - fluctuate :: bool apply random fluctuation to charge.  Default is False.
    '''
    locs = locs.detach().clone()
    squeeze = False
    if len(locs.shape) == 1:
        # make vdim=1 case symmetric with vdim>1 case and undo this on output
        locs = locs[:,None]
        squeeze = True

    npts, vdim = locs.shape;

    if vaxis < 0 or vaxis >= vdim:
        raise ValueError(f'illegal vector axis {vaxis} for vdim={vdim}')

    dt = transport(locs[:,vaxis], target, velocity)
    locs[:,vaxis] = target + torch.zeros_like(locs[:,vaxis])
    if times is None:
        times = dt
    else:
        times = times + dt

    sigma = diffuse(dt, diffusion=diffusion, sigma=sigma)

    default_charge = 1000
    if charge is None:
        charge = torch.zeros(npts, dtype=torch.int32)+default_charge
    charges = absorb(charge, dt, lifetime=lifetime, fluctuate=fluctuate)

    if squeeze:
        locs = torch.squeeze(locs)
    return (locs, times, sigma, charges)
