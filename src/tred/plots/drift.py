#!/usr/bin/env python

import torch
import tred.drift as td

from .util import make_figure

def plot_transport(out):
    target = 0
    velocity = -1
    title = f'drift transport to {target} at velocity {velocity}'
    fig, ax = make_figure(title)
    locs = torch.arange(-1,10)
    dt = td.transport(locs, target, velocity)
    ax.plot(locs, dt)
    ax.set_xlabel("initial location")
    ax.set_ylabel("drift time")
    out.savefig()


def plot_diffuse(out):
    dt = torch.arange(-1, 10)
    D = 1.0
    title = f'drift diffusion for {D=}'
    fig, (ax1,ax2) = make_figure(title, nrows=2, sharex=True)

    sigma1 = td.diffuse(dt, D)
    sigma1[torch.isnan(sigma1)] = 0
    sigma0 = 2
    sigma2 = td.diffuse(dt, D, torch.ones_like(dt)*sigma0)

    ax1.plot(dt, sigma1)
    ax2.plot(dt, sigma2)

    ax1.set_ylabel("sigma")
    ax2.set_ylabel("sigma")
    ax2.set_xlabel("drift time")

    ax1.set_title('initial point like, NaN-clamped')
    ax2.set_title(f'initial finite sigma={sigma0}')

    out.savefig()


def plot_absorb(out):
    dt = torch.arange(-1,10)
    ne0 = 1000
    ne = torch.ones_like(dt) * ne0
    lifetime = 1
    title = f'drift absorb {ne0} electrons {lifetime=}'
    fig, (ax1,ax2) = make_figure(title, nrows=2, sharex=True)

    ne1 = td.absorb(ne, dt, lifetime, fluctuate=False)
    ne2 = td.absorb(ne, dt, lifetime, fluctuate=True)
    
    ax1.plot(dt, ne1)
    ax2.plot(dt, ne2)

    ax1.set_ylabel("ne")
    ax2.set_ylabel("ne")
    ax2.set_xlabel("drift time")

    ax1.set_title('mean number')
    ax2.set_title('fluctuated and clamped')

    out.savefig()

def plots(out):
    plot_transport(out)
    plot_diffuse(out)
    plot_absorb(out)    
