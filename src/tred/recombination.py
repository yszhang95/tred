#!/usr/bin/env python
'''
Functions to apply recombination model, birks or modified box.

Array shapes are described with:

- npts :: the size of a tensor-dimension spanning a set of points (eg depos/steps).
'''
import torch
from torch import Tensor


def birks(dE: Tensor, dEdx: Tensor,
          efield: float, rho: float,
          A3t: float = 0.8, k3t: float = 0.0486,
          Wi:float = 23.6E-6)-> Tensor:
    '''
    Return the number of electrons for a given energy deposition, under birks model.
    Reference: https://lar.bnl.gov/properties/pass.html
    Args:
        dE: 1D tensor of energy deposition, (npt,)
        dEdx: 1D tensor of dE/dx of the energy deposition (npt,)
        efield: electric field, e.g., 0.5 kV/cm
        rho: density of liquid argon, e.g., 1.38 g/cm^3
        A3t: default to 0.8
        k3t: default to 0.0486 (g/MeV cm^2) (kV/cm)
        Wi: W-value for ionization, default to 23.6E-6 MeV/pair
    Return:
        Q: 1D tensor of number of electrons after recombination

    Users are responsible to pass in arguments with consistent units.
    '''
    R = A3t / (1 + dEdx * (k3t / (efield * rho)))
    return R * dE / Wi


def box(dE: Tensor, dEdx: Tensor,
        efield: float, rho: float,
        A: float = 0.93, B: float = 0.212,
        Wi:float = 23.6)-> Tensor:
    '''
    Return the number of electrons for a given energy deposition, under modified box model.
    Reference: https://lar.bnl.gov/properties/pass.html
    Args:
        dE: 1D tensor of energy deposition, (npt,)
        dEdx: 1D tensor of dE/dx of the energy deposition (npt,)
        efield: electric field, e.g., 0.5 kV/cm
        rho: density of liquid argon, e.g., 1.38 g/cm^3
        A: default to 0.93
        B: default to 0.212 (g/MeVcm^2)(kV/cm)
        k3t: default to 0.0486 (g/MeV cm^2) (kV/cm)
        Wi: W-value for ionization, default to 23.6E-6 MeV/pair
    Return:
        Q: 1D tensor of number of electrons after recombination

    Users are responsible to pass in arguments with consistent units.
    '''
    tmp = dEdx * (B / (efield * rho))
    R = torch.log(A + tmp) / tmp
    return R * dE / Wi
