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
    ax2.set_xlabel("drift time (negative is 'backward' drift)")

    ax1.set_title('initial point like')
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

def plot_drift1d(out):
    locs = torch.arange(-1,10)
    (locs, times, sigma, charges) = td.drift(locs, velocity=-1, diffusion=1.0, lifetime=1)
    assert len(locs.shape) == 1
    assert len(sigma.shape) == 1
    assert torch.sum(locs)/len(locs) == locs[0]
    title = 'full drift function - 1D'
    fig, (ax1,ax2) = make_figure(title, nrows=2, sharex=True)
    ax1.plot(times, sigma)
    ax1.set_ylabel('sigma')
    ax2.plot(times, charges)
    ax2.set_ylabel('charge')
    ax2.set_xlabel('drift time')
    out.savefig()

def plot_drift2d(out):
    locx = torch.arange(-1,10)
    locy = torch.arange(0,11)
    locs = torch.vstack((locx,locy)).T
    (locs, times, sigma, charges) = td.drift(locs, velocity=-1, diffusion=torch.tensor([1.0,2.0]), lifetime=1)
    assert len(locs.shape) == 2 # (npts,vdim)
    assert locs.shape[-1] == 2  # vdim
    assert len(sigma.shape) == 2
    assert sigma.shape[-1] == 2    
    assert torch.sum(locs[:0])/locs.shape[0] == locs[0,0]
    title = 'full drift function - 2D'
    fig, (ax1,ax2) = make_figure(title, nrows=2, sharex=True)
    ax1.plot(times, sigma[:,0], label='X')
    ax1.plot(times, sigma[:,1], label='Y')
    ax1.set_ylabel('sigma')
    ax1.legend()
    ax2.plot(times, charges)
    ax2.set_ylabel('charge')
    ax2.set_xlabel('drift time')
    out.savefig()
    

def plots(out):
    plot_transport(out)
    plot_diffuse(out)
    plot_absorb(out)    
    plot_drift1d(out)
    plot_drift2d(out)
