#!/usr/bin/env python
'''
Plots to exercise convo.py.
'''

from .response import get_ndlarsim
from .util import make_figure, SymLogNorm, Ellipse
from tred.convo import dft_shape, zero_pad, convolve
from tred.util import to_tuple, to_tensor
from tred.blocking import Block

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import torch

from collections import namedtuple

def plot_sim1d_time_byhand(out):
    '''
    Simple 1D simulation along time dimension.
    '''
    r = get_ndlarsim()
    res = r[0, 0]               # one path
    dims = (0,)

    sig_t=100
    sig = torch.zeros(sig_t+1)
    sig[sig_t] = 1.0
    c_shape = dft_shape(sig.shape, res.shape)
    print(f'{c_shape=}')

    def dopad(arr, c_shape=c_shape):
        # no rolling on drift dimension
        return zero_pad(arr, c_shape)

    resp = dopad(res)
    sigp = dopad(sig)

    fig, ax = make_figure(f'1D impulse - sig at T={sig_t}')

    sig_spec = torch.fft.fftn(sigp, dim=dims)
    res_spec = torch.fft.fftn(resp, dim=dims)
    meas_spec = sig_spec * res_spec
    meas = torch.fft.ifftn(meas_spec, dim=dims).real

    ax.plot(sigp, label='S')
    ax.plot(resp, label='R')
    ax.plot(meas, label='M')
    ax.set_yscale('log')
    ax.legend()

    out.savefig()

def plot_sim1d_pitch_byhand(out):
    '''
    Simple 1D simulation along pitch dimension.
    '''
    r = get_ndlarsim()
    res = r[0, 0::10, 6310]     # one tick near end
    R = res.shape[0]            # response domain size
    dims = (0,)

    D = 31                      # signal domain size
    sig = torch.zeros(D)
    sig[0] = 1.0
    sig[D//2] = 1.0
    sig[-1] = 1.0
    sig_x = torch.arange(D)

    c_shape = dft_shape(sig.shape, res.shape)
    M = c_shape[0]              # measurement domain size

    print(f'{sig.shape} * {res.shape} -> {c_shape=}')

    def do_res_pad(arr, c_shape=c_shape):
        o_shape = to_tensor(arr.shape, dtype=torch.int32)
        h_shape = o_shape // 2
        rolled = torch.roll(arr, shifts=to_tuple(h_shape), dims=dims)
        padded = zero_pad(rolled, c_shape)
        return torch.roll(padded, shifts=to_tuple(-h_shape), dims=dims), h_shape.item()

    def do_sig_pad(arr, c_shape=c_shape):
        padded = zero_pad(arr, c_shape)
        h_shape = (to_tensor(c_shape, dtype=torch.int32) - to_tensor(arr.shape, dtype=torch.int32))//2
        print(f'{h_shape=}')
        return torch.roll(padded, shifts=to_tuple(h_shape), dims=dims), h_shape.item()

    resp, r_half = do_res_pad(res)
    sigp, s_half = do_sig_pad(sig, c_shape)
    sigp_x = torch.arange(-s_half, -s_half+c_shape[0])
    print(f'{sigp.shape=} {sigp_x.shape}')

    print(f'{r_half=} {s_half=}')

    sig_spec = torch.fft.fftn(sigp, dim=dims)
    res_spec = torch.fft.fftn(resp, dim=dims)
    meas_spec = sig_spec * res_spec
    meas = torch.fft.ifftn(meas_spec, dim=dims).real

    nconv = f'n={c_shape[0]}={sig.shape[0]}+{res.shape[0]}-1'

    # [sig, blank, res]
    # [sigp, blank, resp]
    # [blank, meas, blank]
    fig = plt.figure(layout='constrained')
    gs = GridSpec(3, 4, figure=fig)

    # signal
    ax = fig.add_subplot(gs[0,0:2])
    ax.set_title(f'Signal n={sig.shape[0]}')
    ax.plot(sig_x, sig, color='blue', label='signal', marker='o', linestyle='none')
    ax.set_xlim(-s_half, -s_half + c_shape[0])
    axs = ax

    ax = fig.add_subplot(gs[1,0:2])
    ax.set_title(f'Edge-padded {nconv}')
    ax.plot(sigp_x, sigp, color='blue', label='padded', marker='o', linestyle='none')
    ax.set_xlim(-s_half, -s_half + c_shape[0])
    axsp = ax

    con = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data",
                          axesA=axs, axesB=axsp, color="blue")
    axsp.add_artist(con)
    con = ConnectionPatch(xyA=(sig.shape[0],0), xyB=(sig.shape[0],0), coordsA="data", coordsB="data",
                          axesA=axs, axesB=axsp, color="blue")
    axsp.add_artist(con)

    # response
    ax = fig.add_subplot(gs[0,2:4])
    ax.set_title(f"Response n={res.shape[0]}")
    ax.plot(res, color='red', label='response', drawstyle='steps-mid')
    axr = ax
    
    ax = fig.add_subplot(gs[1,2:4])
    ax.set_title(f"Center-padded {nconv}")
    ax.plot(resp, color='red', label='padded', drawstyle='steps-mid')
    axrp = ax

    con = ConnectionPatch(xyA=(r_half,0), xyB=(r_half,0), coordsA="data", coordsB="data",
                          axesA=axr, axesB=axrp, color="red")
    axrp.add_artist(con)
    con = ConnectionPatch(xyA=(r_half,0), xyB=(r_half+sig.shape[0],0), coordsA="data", coordsB="data",
                          axesA=axr, axesB=axrp, color="red")
    axrp.add_artist(con)



    # measure
    ax = fig.add_subplot(gs[2,1:3])
    ax.set_title(f'Convolution measure {nconv}')
    ax.plot(meas, label='measured', color='purple', drawstyle='steps-mid')
    axm = ax
    
    


    out.savefig()


def plot_sim2d_byhand(out):
    '''
    Do a simple 2D sim, unbatched
    '''
    r = get_ndlarsim()
    res = r[0, 0::10]
    dims = (0,1)

    sig_z=2
    sig_t=10

    sig = torch.zeros((11,30))
    sig[sig_z, sig_t] = 1.0

    c_shape = dft_shape(sig.shape, res.shape)
    print(f'{c_shape=}')
    
    def do_res_pad(arr, c_shape=c_shape):
        # note, when batched we must change this code.
        o_shape = to_tensor(arr.shape, dtype=torch.int32)
        h_shape = o_shape // 2
        h_shape[-1] = 0         # t dimension is "causal", no roll
        rolled = torch.roll(arr, shifts=to_tuple(h_shape), dims=dims)
        padded = zero_pad(rolled, c_shape)
        return torch.roll(padded, shifts=to_tuple(-h_shape), dims=dims), h_shape

    def do_sig_pad(arr, c_shape=c_shape):
        padded = zero_pad(arr, c_shape)
        h_shape = (to_tensor(c_shape, dtype=torch.int32) - to_tensor(arr.shape, dtype=torch.int32))//2
        h_shape[-1] = 0         # t dimension is "causal", no roll
        return torch.roll(padded, shifts=to_tuple(h_shape), dims=dims), h_shape


    resp, r_half = do_res_pad(res)
    sigp, s_half = do_sig_pad(sig)

    fig, (ax1,ax2) = make_figure('2D impulse', nrows=2)

    def imshow(ax, thing):
        imnorm = SymLogNorm(linthresh=0.01, vmin=thing.min(), vmax=thing.max())
        im = ax.imshow(thing[:,6000:], aspect='auto', interpolation='none', norm=imnorm)
        fig.colorbar(im)

    imshow(ax1, resp)
    ax1.plot([0,400],[sig_z,sig_z], color="red")


    # ... convolution ...

    sig_spec = torch.fft.fftn(sigp, dim=(0,1))
    res_spec = torch.fft.fftn(resp, dim=(0,1))
    meas_spec = sig_spec * res_spec
    meas = torch.fft.ifftn(meas_spec, dim=(0,1)).real

    imshow(ax2, meas)
    ax2.plot([0,400],[sig_z+s_half[0],sig_z+s_half[0]], color="red")

    out.savefig()





def plot_simNd(out):
    '''
    Iterate over many simulation dimensions using tred
    '''

    alls = slice(0, None)       # slice over entire dimension
    imps = slice(0, None, 10)   # slice over first impact
    
    sig_t_size = 100
    sig_p_size = 31

    Params = namedtuple("Params", "rslice dims sshape imps")
    variants = [
        # 1D over time
        Params(rslice=(0,0,alls),
               dims=(0,),
               imps=[(0,), (sig_t_size//2,), (sig_t_size-1,)],
               sshape=(sig_t_size,)),
             
        # 1D over pitch - lack of time/drift dimension is unsupported by convolve()
        # Params(rslice=(0, imps, 6310),
        #        dims=(0,),
        #        imps=[(0,), (sig_p_size//2,), (sig_p_size-1,)],
        #        sshape=(sig_p_size,)),
             
        # 2D
        Params(rslice=(0, imps, alls),
               dims=(0,1),
               imps=[(0,0), (sig_p_size//2,sig_t_size//2), (sig_p_size-1,sig_t_size-1)],
               sshape=(sig_p_size, sig_t_size)),
        
        # 3D
        Params(rslice=(imps, imps, alls),
               dims=(0,1,2),
               imps=[(0,0,0), (sig_p_size//2,sig_p_size//2,sig_t_size//2), (sig_p_size-1,sig_p_size-1,sig_t_size-1)],
               sshape=(sig_p_size, sig_p_size, sig_t_size))
    ]


    res_raw = get_ndlarsim()
    print(f'{res_raw.shape=}')
    for p in variants:
        print(p)
        res = res_raw[p.rslice]
        print(f'{res.shape=}')
        
        sig_data = torch.zeros(p.sshape)
        for imp in p.imps:
            sig_data[imp] = 1

        ndim = len(p.dims)
        loc = to_tensor([0]*ndim)
        sig = Block(location=loc, data=sig_data)

        c_shape = dft_shape(sig.shape, res.shape)
        print(f'{sig.shape=} (x) {res.shape=} = {c_shape=}')

        meas = convolve(sig, res)
        print(f'{meas.location=} {meas.shape=} {meas.data.shape=}')


        if ndim == 1:
            fig, ax = make_figure('1D impulses')
            ax.plot(meas.data[0,6000:])

        if ndim == 2:
            fig, ax = make_figure('2D impulses')
            ax.imshow(meas.data[0,:,6000:], interpolation='none', aspect='auto')

        if ndim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            dat = meas.data[0]
            vmin = dat.min()
            vmax = dat.max()
            amm = torch.abs(dat).max()
            dat[torch.abs(dat) < amm/100.0] = 0
            ind = dat.nonzero()
            print(f'{ind.shape=}')
            i,j,k = ind[:,0], ind[:,1], ind[:,2]
            val = dat[i, j, k]
            print(f'{len(val)}')
            ax.scatter(i, j, k, c=(val+vmin)/(vmin+vmax), alpha=torch.abs(val)/amm)

        out.savefig()
            


def plots(out):
    plot_sim1d_time_byhand(out)
    plot_sim1d_pitch_byhand(out)
    plot_sim2d_byhand(out)
    
    plot_simNd(out)
