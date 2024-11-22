#!/usr/bin/env python
'''
Functions to apply drift models.
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
    '''

    if fluctuate:
        loss = 1.0 - torch.exp(-dt / lifetime)
        loss = torch.clamp(loss, 0, 1, out=loss)
        b = Binomial(ne, loss)
        return ne - b.sample()

    return ne * torch.exp(-dt / lifetime)
