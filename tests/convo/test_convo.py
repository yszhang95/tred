import torch
from tred.convo import symmetric_pad, convolve
from tred.blocking import Block
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

def plot_convolve_2d():
    # Needs to include an extra dimension due to annoying setup of padding in convo.py. .
    q = torch.tensor([[1,0], [0,1]]) # non-batched, unit charge;
    response = torch.tensor([[1,0], [2,0], [1,0]]) # 3 pixels; nothing in the next moment

    response_conv = torch.tensor([[2,0],[1,0],[1,0]])

    signal = Block(location=torch.tensor([[0,0]]), data=q)

    result = convolve(signal, response_conv)
    X = result.data[0].numpy().T
    loc = result.location[0][[1,0]]

    x = np.arange(loc[1], loc[1]+X.shape[1], 1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8*2, 8*2))
    axes = axes.flat
    for i in range(X.shape[0]):
        axes[i].bar(x, height=X[i])
        axes[i].set_title(f'time == {loc[0]+i}')
        # Force integer ticks on y-axis
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Force integer ticks on x-axis (if needed)
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add the text inside the plot, anchored to the top-left (axes coordinates: (0, 1))
    axes[3].text(
        0.2, 0.8,  # x, y position in axes fraction (0 = left/bottom, 1 = right/top)
        f"Initial location on lower-left corner {signal.location[0].tolist()}\n"
        f"Initial Q {signal.data[0].tolist()}\n"
        f"taxis=-1\n"
        f"Response {response_conv.tolist()}",
        fontsize=12,
        va='top',   # vertical alignment
        ha='left',  # horizontal alignment
        transform=axes[3].transAxes,  # so (0,1) refers to axes coordinates, not data
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')  # optional, to make it more readable
    )

    fig.savefig('unitq_convolve_2d.png')


if __name__ == '__main__':
    plot_symmetric_pad_1d()
    plot_sr_pad_1d()
    plot_convolve_2d()
