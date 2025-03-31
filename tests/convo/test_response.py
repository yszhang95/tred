from tred.response import Response, quadrant_copy, ndlarsim

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_quadrant_copy():
    wf = torch.arange(2*4*4).reshape(2,4,4)
    wf = quadrant_copy(wf, axis=0, even=True)
    wf = Response(current=wf, spacing=(1,1,1), axis=0)
    wf = wf.current.numpy()

    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, figsize=(8*2, 8))

    for i in range(len(wf)):
        im = axes[i].imshow(wf[i], cmap=cmap)
        axes[i].set_title(f"wf[{i}]")
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], ticks=range(2*4*4-1), orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig('quadrant_copy.png')

def plot_ndlar_response():
    response_path = '/home/yousen/Documents/NDLAr2x2/tred/response_38_v2b_50ns_ndlar.npy'
    res_np = np.load(response_path)
    response = ndlarsim(response_path)

    X = response[:,:,np.argmax(res_np[0,0])].numpy()

    fig, ax = plt.subplots()
    cax = ax.matshow(X)

    # Set ticks at intervals of 10
    xticks = range(0, X.shape[0], 10)
    yticks = range(0, X.shape[1], 10)
    ax.set_xticks(yticks)
    ax.set_yticks(xticks)

    # Enable grid on top of the image
    ax.grid(which='both', color='white', linestyle='-', linewidth=0.5)

    ax.set_xlabel('index along axis 1')
    ax.set_ylabel('index along axis 0')

    ax.set_title('when current at the center of collection pixel reaches maximum')

    fig.colorbar(cax)

    fig.savefig('ndlar_response_peak_at_pxlctr.png')

def main():
    plot_quadrant_copy()
    plot_ndlar_response()

if __name__ == '__main__':
    main()
