import torch
from tred.convo import symmetric_pad, convolve
from tred.blocking import Block
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import conv2d, pad

def plot_symmetric_pad_1d():
    # (odd, even) and (odd, even)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8*2, 8*2))
    for i, nt in enumerate(range(4, 6)):
        for j, nz in enumerate(range(3,5)):
            t = torch.arange(nt) + 1
            shape = (nt+nz,)
            for k, style in enumerate(['append', 'prepend', 'center', 'edge']):
                tp = symmetric_pad(t, shape, (style,))
                axes[i,j].stem(np.arange(shape[0])+0.05*k, tp.numpy(), linefmt=f'C{k}-', basefmt=" ", label=style)
            axes[i,j].legend()
            axes[i,j].axhline(0, color='gray', linestyle='--', linewidth=1)
            axes[i,j].set_title(f'nt={nt}, nz={nz}')

    fig.savefig('symmetric_pad_1d.png')

def plot_sr_pad_1d():
    # (odd, even) and (odd, even)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8*2, 8*2))
    for i, ns in enumerate(range(4, 6)):
        for j, nr in enumerate(range(3,5)):
            s = 2*torch.ones((ns,))
            r = 4*torch.ones((nr,))
            shape = (ns+nr-1,)
            sp = symmetric_pad(s, shape, ('edge',))
            rp = symmetric_pad(r, shape, ('center',))
            axes[i,j].stem(np.arange(shape[0])-0.05, sp.numpy(), linefmt=f'C{0}-', basefmt=" ", label='edge')
            axes[i,j].stem(np.arange(shape[0])+0.05, rp.numpy(), linefmt=f'C{4}-', basefmt=" ", label='center')
            axes[i,j].legend()
            axes[i,j].axhline(0, color='gray', linestyle='--', linewidth=1)
            axes[i,j].set_title(f'ns={ns}, nr={nr}, shape={shape[0]}')

    fig.savefig('sr_pad_1d.png')



if __name__ == '__main__':
    plot_symmetric_pad_1d()
    plot_sr_pad_1d()
