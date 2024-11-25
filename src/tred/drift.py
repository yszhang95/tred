#!/usr/bin/env python
'''
Functions to apply drift models.

Caveat: all functions assume 1D, caller should slice accordingly.

'''

import torch
from torch.distributions.binomial import Binomial

def transport(locs, target, velocity):
    '''
    Return times to transport points at locs to location target at given speed.

    All inputs are signed.  Return transport times are negative when the
    transport is in the opposite direction of velocity.

    - locs :: real 1D array of locations of points on the axis.
    - target :: real scalar location on the axis.
    - velocity :: real scalar speed of transport along the axis.

    Caveat: 1D space only, caller slice accordingly.
    '''
    return (target - locs)/velocity


def diffuse(dt, D, sigma=None):
    '''
    Return new Gaussian diffusion sigma after drifting by time dt.

    - dt :: real 1D tensor of drift times
    - D :: real scalar diffusion coefficient
    - sigma :: real 1D tensor giving initial Gaussian widths or None if points

    Note, negative dt entries will result in shrinking of initial nonzero sigma.
    If this shrinking goes below zero then NaN is resulting.

    If this is unwanted, exclude negative values.  Eg,

    sigmaL = zeros_like(dt)
    fwd = dt>0
    sigmaL[fwd] = diffuse(dt[fwd], DL)

    Or, to clamp to zero:

    sigmaL = diffuse(dt, DL)
    sigmaL[torch.isnan(sigmaL)] = 0

    Caveat: 1D space only, caller slice accordingly.
    '''
    if sigma is None:
        return torch.sqrt(2*D*dt)
    return torch.sqrt(2*D*dt + sigma*sigma)


def absorb(ne, dt, lifetime, fluctuate=False):
    '''
    Apply an absorption lifetime return counts of surviving electrons.

    - ne :: positive integer 1D tensor giving number of electrons at initial points.
    - dt :: real 1D tensor of time each point is subject to absorption.
    - lifetime :: scalar value giving exponential lifetime
    - fluctuate :: if true, apply binomial fluctuation based on mean lost q

    Note, negative dt will lead to exponential increase not decrease.

    Caveat: 1D space only, caller slice accordingly.
    '''

    if fluctuate:
        loss = 1.0 - torch.exp(-dt / lifetime)
        loss = torch.clamp(loss, 0, 1, out=loss)
        b = Binomial(ne, loss)
        return ne - b.sample()

    return ne * torch.exp(-dt / lifetime)


def drift(locs, target, velocity, D, lifetime, charge=None, sigma=None, fluctuate=False):
    '''
    Full drift() = transport() + diffuse() + absorb().

    Returns tuple: (drifted, sigma, charges)

    See docs on individual functions for other arguments.

    Caveat: 1D space only, caller slice accordingly.
    '''
    drifted = transport(locs, target, velocity)

    sigma = diffuse(drifted, D=D, sigma=sigma)
    sigma[torch.isnan(sigma)] = 0

    default_charge = 1000
    if charge is None:
        charge = torch.zeros(locs.shape[0], dtype=torch.int32)+default_charge
    charges = absorb(charge, drifted, lifetime=lifetime, fluctuate=fluctuate)
    return (drifted, sigma, charges)
