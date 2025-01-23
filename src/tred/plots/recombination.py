#!/usr/bin/env python

import torch
import tred.recombination as td

from .util import make_figure

def plot_recombination(out):
    efield = 0.5 # kV/cm
    rho = 1.38 # g/cm^3
    A3t = 0.8 # birks
    k3t = 0.0486 # (g/MeV cm^2) (kV/cm); birks
    A = 0.93 # box
    B = 0.212 # (g/MeVcm^2)(kV/cm); box
    Wi = 23.6E-6 # MeV/pair
    dEdx = torch.linspace(0.4, 32, 500) # MeV/cm
    dx = 3 # cm
    dE = dEdx * dx
    Qbirks = td.birks(
        dE, dEdx, efield, rho, A3t, k3t, Wi
    )
    Qbox = td.box(
        dE, dEdx, efield, rho, A, B, Wi
    )
    dQdx_birks = Qbirks/dx
    dQdx_box = Qbox/dx

    r_birks = dQdx_birks / dEdx * Wi
    r_box = dQdx_box / dEdx * Wi

    title = 'Comparison between birks and modified box models'
    fig, (ax0, ax1) = make_figure(title, nrows=1, ncols=2)
    ax0.plot(dEdx, r_birks, label='Birks model')
    ax0.plot(dEdx, r_box, label='Modified box model')
    # ax0.plot(dEdx, dQdx_birks, label='Birks model')
    # ax0.plot(dEdx, dQdx_box, label='Birks model')

    ax0.set_xlabel('dE/dx [MeV/cm]')
    ax0.set_ylabel(f'W_i * dQ/dx / dE/dx @ E={efield} kV/cm, LAr density={rho} g/cm^3')
    ax0.legend()

    ax1.text(0.2, 0.8, f'Birks: A3t {A3t}')
    ax1.text(0.2, 0.7, f'Birks: k3t {k3t} (g/MeV cm^2) (kV/cm)')
    ax1.text(0.2, 0.6, f'Box: A {A}')
    ax1.text(0.2, 0.5, f'Box: B {B} (g/MeV cm^2) (kV/cm)')
    ax1.text(0.2, 0.4, f'Wi: {Wi} MeV/pair')
    ax1.text(0.2, 0.3, f'dx = {dx} cm')

    out.savefig()

def plots(out):
    plot_recombination(out)
