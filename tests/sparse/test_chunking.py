'''
Additional tests for `tests/test_chunking.py`.
'''

import tred
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Generate N visually distinct colors using HSV
def generate_colors(n):
    colors = plt.cm.hsv(np.linspace(0, 1, n))
    return colors

from tred.chunking import location as chunking_location
from tred.chunking import content as chunking_content
from tred.chunking import accumulate as chunking_accumulate
from tred.blocking import Block

def box2d(c0, c1):
    return np.array([[c0[0], c0[0], c1[0], c1[0], c0[0]],
                     [c0[1], c1[1], c1[1], c0[1], c0[1]]])

def plot_2dloc(loc, shape, figname='fig.png'):
    fig, ax = plt.subplots()
    for i in range(len(loc)):
        pts = box2d(loc[i].numpy(), (loc[i] + shape).numpy())
        ax.plot(pts[0], pts[1], 'o-')
    plt.savefig(figname)

def plot_2dblock(block: Block, figname='fig.png'):
    shape = block.shape
    loc = block.location
    plot_2dloc(loc, shape, figname)

def plot_grid_on_axis(ax, c0, c1, grid_values):
    """
    Plot the box outline and annotate every integer grid point within the box on the given axis.

    Parameters:
      ax         : Matplotlib axis to plot on.
      c0, c1     : Two-element lists/arrays defining the lower-left and upper-right corners.
      grid_values: A 2D numpy array containing the values to annotate at each grid point.
                   Its shape should be ((c1[1]-c0[1]+1), (c1[0]-c0[0]+1)).
    """
    # Create grid coordinates (inclusive)
    x = np.arange(c0[0], c1[0] - 0.1)
    y = np.arange(c0[1], c1[1] - 0.1)
    xx, yy = np.meshgrid(x, y)

    # Plot the box outline
    pts = box2d(c0, c1)
    ax.plot(pts[0], pts[1], 'k-')

    # Loop through grid points to plot a marker and annotate with grid_values
    for i in range(len(y)):
        for j in range(len(x)):
            ax.plot(xx[i, j], yy[i, j], 'ro')
            ax.text(xx[i, j]+0.1, yy[i, j]+0.1, str(grid_values[i, j]),
                    fontsize=10, color='black', ha='center', va='center')

    # Set limits and grid details

def plot_batch_grids(c0_list, c1_list, grid_values_list, shape=(2,2), figname='batch_grid.png'):
    nbatch = c0_list.shape[0]
    nchunks = c0_list.shape[1]
    fig, axs = plt.subplots(shape[0], shape[1], figsize=(8, 8))
    if shape[0] + shape[1] > 2:
        axs = axs.flatten()  # flatten for easy iteration
    else:
        axs = [axs]

    # Plot each box on a subplot axis
    for i in range(nbatch):
        for j in range(nchunks):
            plot_grid_on_axis(axs[i], c0_list[i,j], c1_list[i,j], grid_values_list[i,j])
        axs[i].grid(True)
        axs[i].set_title(f'ibatch{i}')
        # ax.legend()
    xlims = np.array([axs[i].get_xlim() for i in range(nbatch)])
    ylims = np.array([axs[i].get_ylim() for i in range(nbatch)])

    for i in range(nbatch):
        axs[i].set_xlim((np.min(xlims), np.max(xlims)))
        axs[i].set_ylim((np.min(ylims), np.max(ylims)))

    # Remove any extra subplots if nbatch < total axes available.
    for j in range(nbatch, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def plot_chunking_location_2d():
    data = torch.rand((3,2,2))
    # loc = torch.randint(0, 10, (3,2))
    loc = torch.tensor([[6, 10],
        [5, 9],
        [3, 8]])
    block = Block(location=loc, data=data)

    plot_2dblock(block, figname='block2d.png')
    plot_2dloc(chunking_location(block, chunk_shape=(1,1)).view(-1,2), torch.tensor([1,1]),
               'chunked_block2d.png')

def plot_chunking_2d():
    nbatch = 3
    cshape = (2, 2)
    mshape = (2, 3)
    dshape = [cshape[i] * mshape[i] for i in range(2)]

    dsize = [nbatch,] + dshape

    data = np.arange(nbatch)
    data = data.repeat(np.prod(dshape)).reshape(*dsize)
    # loc = torch.randint(0, 10, (3,2))
    loc = torch.tensor([[6, 10],
                        [4, 6], [4, 8]])
    block = Block(location=loc, data=torch.tensor(data))

    locs = chunking_location(block, chunk_shape=cshape)
    c0 = locs.reshape(nbatch, -1, 2).numpy()
    c1 = c0 + np.array(cshape).reshape(1, 1, 2)

    contents = chunking_content(block, cshape)

    accumulated_data = chunking_accumulate(Block(data=contents.reshape(-1, 2, 2), location=locs.reshape(-1, 2)))

    contents = contents.reshape(nbatch, -1, 2, 2)

    plot_batch_grids(c0, c1, contents.numpy(), figname='chunked_content.png')

    c0acc = accumulated_data.location.view(1, -1, 2).numpy()
    c1acc = c0acc + np.array(cshape).reshape(1,1,2)
    contentacc = accumulated_data.data.view(1, -1, 2, 2).numpy()

    plot_batch_grids(c0acc, c1acc, contentacc, shape=(1,1), figname='chunked_sum.png')

def test_chunking_accumulate():
    '''
    Tested values based on plot_chunking_2d()
    '''
    target_locations = np.array([(i,j) for i in range(4, 8, 2) for j in range(6, 14, 2)]).reshape(2, 4, 2)
    target_data = np.tile(np.array((1,3,3,2)).repeat(2), (4,1))
    target_data = target_data.reshape(2,2,8).transpose(0,2,1).reshape(2,4,2,2)
    # print(target_locations[0], target_locations[1])
    # print(len(target_data), len(target_locations))

    nbatch = 3
    cshape = (2, 2)
    mshape = (2, 3)
    dshape = [cshape[i] * mshape[i] for i in range(2)]

    dsize = [nbatch,] + dshape

    data = np.arange(nbatch)
    data = data.repeat(np.prod(dshape)).reshape(*dsize)
    # loc = torch.randint(0, 10, (3,2))
    loc = torch.tensor([[6, 10],
                        [4, 6], [4, 8]])
    block = Block(location=loc, data=torch.tensor(data))

    locs = chunking_location(block, chunk_shape=cshape)

    contents = chunking_content(block, cshape)

    accumulated_data = chunking_accumulate(Block(data=contents.reshape(-1, 2, 2), location=locs.reshape(-1, 2)))

    location = accumulated_data.location.numpy()
    data = accumulated_data.data.numpy()

    idxs = np.argsort(location[:,0]*100 + location[:,1])

    assert np.all(location[idxs].reshape(2,4,2) == target_locations)
    assert np.all(data[idxs].reshape(2,4,2,2) == target_data)


def main():
    plot_chunking_location_2d()
    plot_chunking_2d()
    test_chunking_accumulate()


if __name__ == '__main__':
    main()
