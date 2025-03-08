#!/usr/bin/env python
'''
Functions to apply drift related models.

Array shapes are described with:

- npts :: the size of a tensor-dimension spanning a set of points (eg depos/steps).
- vdim :: the size of a tensor-dimension spanning spatial vector-dimensions.

Eg: (npts=5,vdim=3) is 5 points in 3-space.

'''

from .util import debug, tenstr, to_tensor

import torch
from torch.distributions.binomial import Binomial


def transport(locs, target, velocity):
    '''
    Return the time to drift from initial locations (locs) on some axis to a
    target location on that axis at the given (signed) velocity.

    - locs :: real 1D tensor (npts,)
    - target :: real scalar
    - velocity :: real scalar
    '''
    return (target - locs)/velocity


def diffuse(dt, diffusion, sigma=None):
    '''
    Return new Gaussian diffusion sigma after drifting by time dt.

    - dt :: scalar or real 1D tensor of drift times (npts,).
    - diffusion :: real scalar (number or 0D tensor) or 1D (vdim,) tensor of diffusion coefficients.
    - sigma :: real 1D (npts) or 2D (npts,vdim) tensor or None.

    A sigma of None indicate no prior diffusion.

    Note, negative dt entries will result in a shrinking of any initial sigma.
    Where this shrinkage may go negative the resulting sigma element is set to
    zero.
    '''
    squeeze = False

    # eg, diffusion is 5; it cannot be a list/tuple
    if not isinstance(diffusion, torch.Tensor):
        diffusion = torch.tensor([diffusion], device=dt.device)
        squeeze = True

    # eg, diffusion is a 0D tensor
    if diffusion.dim() == 0:
        diffusion = diffusion.unsqueeze(0)
        squeeze = True

    if len(dt.shape) != 1:
        raise ValueError(f'unsupported shape for dt: {dt.shape}')

    if len(diffusion.shape) != 1:
        raise ValueError(f'unsupported shape for diffusion: {diffusion.shape}')

    vdim = len(diffusion.shape)
    if sigma is not None and len(sigma.shape) != vdim:
        raise ValueError(f'shape of sigma ({sigma.shape}) does not span {vdim} dimensions')

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
    This is unphysical. The surviving electrons (charge) remain the same when dt is negative.
    No negative binomial fluctuations or any factor from the exponential distribution is applied.
    '''
    charge = charge.to(dtype=torch.int32)
    if fluctuate:
        loss = 1.0 - torch.exp(-dt / lifetime)
        loss = torch.clamp(loss, 0, 1, out=loss)
        b = Binomial(charge, loss)
        return charge - b.sample()
    return torch.where(dt>=0, charge * torch.exp(-dt / lifetime), charge)


def drift(locs, velocity, diffusion, lifetime, target=0,
          times=None, vaxis=0, charge=None, sigma=None, fluctuate=False,
          tshift=None, drtoa=None):
    '''
    Apply drift models.

    Returns tuple of post-drift quantities:

      (locs, times, sigma, charges)

    - locs :: 1D (npts, ) or 2D (npts, vdim) tensor of the updated locs. The original locs is kept except along vaxis.
              locs along vaxis is updated to target.
    - times :: tensor with the same shape as locs's. It is
               initial time (the argument times) + dt (from transport) - tshift (given
               or from abs(drtoa/velocity), drtoa = abs(loc_anode - loc_respone)).
    - sigma :: real 1D (npts,) or 2D (npts, vdim) tensors by adding the initial sigma and diffusion width in quadrature.
    - charges :: quenched charges by function absorb.

    Note: sigma, charges, locs are post-drift quantities at anode planes. The value of times depends on the potential
          shifts from tshift or drtoa, may not be time when survivial electrons arrive at anodes, but the value pulled
          back to response plane.

    Required arguments:

    - locs :: 1D (npts,) or 2D (npts,vdim) tensor of initial locations.
    - velocity :: real scalar (signed) velocity from initial to final location.
    - diffusion :: real scalar or 1D (vdim,) diffusion coefficient.
    - lifetime :: positive real scalar electron absorption lifetime

    Optional arguments:

    - target :: real scalar, location on vaxis to cease drift, i.e., anode plane.  Default is 0.
    - times :: 1D (npts,) tensor of initial times.  Default is zeros.
    - vaxis :: int scalar.  The vector-axis on which to drift.  Default is 0.
    - charge :: real scalar or 1D (npts,) initial charge.  Default is 1000 electrons.
    - sigma :: real 1D (npts,) or 2D (npts,vdim) tensor giving initial sigma.  Default is none.
    - fluctuate :: bool apply random fluctuation to charge.  Default is False.
    - tshift :: real positive scalar (number or 0D tensor), to correct for drift time from response plane to anode.
                Note: users must be responsible for the correct shift in use.
    - drtoa :: real positive scalar (number or 0D tensor). The distance along drift direction from response plane
               to anode plane, abs(loc_anode - loc_response). It is mutually exclusive with tshift. Default to None.
               Note: users must be responsible for the consistency between drift velocity in TRED
               and in field response calculations.

    FIXME: units of drtoa must be consistent with locs.
    '''
    if tshift is not None and drtoa is not None:
        raise ValueError('tshift and drtoa are mutually exclusive arguments.')

    locs = locs.detach().clone()
    squeeze = False
    if len(locs.shape) == 1:
        # make vdim=1 case symmetric with vdim>1 case and undo this on output
        locs = locs[:,None]
        squeeze = True

    npts, vdim = locs.shape

    if vdim > 1 and len(diffusion.shape) != 1:
        raise ValueError(f'illegal shape for diffusion coefficients: {diffusion.shape}, expect 1D')

    if sigma is not None:
        if len(sigma.shape) != vdim:
            raise ValueError(f'diffusion sigma shape {sigma.shape} does not span {vdim} dimensions')

    if vaxis < 0 or vaxis >= vdim:
        raise ValueError(f'illegal vector axis {vaxis} for vdim={vdim}')

    dt = transport(locs[:,vaxis], target, velocity)
    locs[:,vaxis] = target + torch.zeros_like(locs[:,vaxis])

    if times is None:
        times = dt
    else:
        times = times + dt

    if drtoa is not None:
        tshift = drtoa / velocity
    if tshift is not None:
        tshift = tshift if isinstance(tshift, torch.Tensor) else to_tensor(tshift, dtype=torch.float32, device=times.device)
    if tshift is not None:
        times = times - torch.abs(tshift)

    debug(f'dt:{tenstr(dt)} diffusion:{tenstr(diffusion) if isinstance(diffusion, torch.Tensor) else diffusion}')
    sigma = diffuse(dt, diffusion=diffusion, sigma=sigma)

    default_charge = 1000
    if charge is None:
        charge = torch.zeros(npts, dtype=torch.int32, device=locs.device)+default_charge
    charges = absorb(charge, dt, lifetime=lifetime, fluctuate=fluctuate)

    if squeeze:
        locs = torch.squeeze(locs)
    return (locs, times, sigma, charges)
