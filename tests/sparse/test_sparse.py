import torch
import pytest
from tred.sparse import SGrid  # Replace with the actual module name where SGrid is defined
from tred.blocking import Block  # Replace with the actual module path
from tred.sparse import chunkify
from tred.graph import ChunkSum

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def test_vdim():
    # Test that the number of spatial dimensions equals the length of the spacing vector.
    spacing = torch.tensor([2, 3, 4])
    grid = SGrid(spacing)
    assert grid.vdim == 3

def test_to_tensor():
    # Test that to_tensor converts a list to a torch.Tensor on the same device as spacing.
    spacing = torch.tensor([2, 2])
    grid = SGrid(spacing)
    data = [1, 2, 3]
    tensor_data = grid.to_tensor(data)
    assert isinstance(tensor_data, torch.Tensor)
    assert tensor_data.device == spacing.device

def test_spoint_and_gpoint():
    # For spacing [2,2], a grid point [3,4] should map to super-grid index [1,2].
    spacing = torch.tensor([2, 2])
    grid = SGrid(spacing)

    # Non-batched input without offset.
    gpt = torch.tensor([3, 4])
    sp = grid.spoint(gpt)
    expected_sp = torch.tensor([1, 2])
    assert torch.equal(sp, expected_sp)

    # gpoint should convert the super point back to grid coordinates (multiplied by spacing).
    gpt_from_sp = grid.gpoint(sp)
    expected_gpt_from_sp = torch.tensor([2, 4])
    assert torch.equal(gpt_from_sp, expected_gpt_from_sp)

    # Test with offset provided.
    offset = torch.tensor([1, 1])
    sp_offset = grid.spoint(gpt, goffset=offset)
    # Calculation: ([3,4] - [1,1]) // [2,2] = [2,3] // [2,2] = [1,1]
    expected_sp_offset = torch.tensor([1, 1])
    assert torch.equal(sp_offset, expected_sp_offset)

def test_batched_spoint():
    # Test that batched input (2D tensor) for spoint returns the correct batched result.
    spacing = torch.tensor([2, 2])
    grid = SGrid(spacing)

    # Batched grid points.
    gpts = torch.tensor([[3, 4],
                           [5, 6]])
    sp = grid.spoint(gpts)
    # Expected:
    #   Row 1: [3//2, 4//2] = [1, 2]
    #   Row 2: [5//2, 6//2] = [2, 3]
    expected = torch.tensor([[1, 2],
                             [2, 3]])
    assert torch.equal(sp, expected)

def test_envelope():
    # For a given Block (pbounds) with location [1,1] and shape [3,4]
    # and spacing [2,2], the envelope should cover:
    #   minpts = gpoint(spoint([1,1])) = [0,0]    (since 1//2 = 0)
    #   maxpts = gpoint(spoint([3,4]) + 1)
    #          = gpoint(([3//2,4//2] + 1)) = gpoint([1+1,2+1]) = gpoint([2,3])
    #          = [2*2, 3*2] = [4,6]
    # So the envelope should have location [0,0] and shape [4,6].
    spacing = torch.tensor([2, 2])
    grid = SGrid(spacing)

    # Create a Block with location and shape.
    pbounds = Block(location=torch.tensor([[1, 1], [2, 3]]), shape=torch.tensor([3, 4]))
    env = grid.envelope(pbounds)

    expected_location = torch.tensor([[0, 0], [2, 2]])  # Block stores location as a batched tensor.
    expected_shape = torch.tensor([4, 6])
    assert torch.equal(env.location, expected_location), f'envelope location {env.location}, expected_location {expected_location}'
    assert torch.equal(env.shape, expected_shape), f'envelope shape {env.shape}, expected_shape {expected_shape}'

# Generate N visually distinct colors using HSV
def generate_colors(n):
    colors = plt.cm.hsv(np.linspace(0, 1, n))
    return colors

def box2d(c0, c1):
    return np.array([[c0[0], c0[0], c1[0], c1[0], c0[0]],
                     [c0[1], c1[1], c1[1], c0[1], c0[1]]])

def plot_grid_on_axis(ax, c0, c1, grid_values, offset=(0.1, 0.1), **kwargs):
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
    ax.plot(pts[0], pts[1], **kwargs)

    # Loop through grid points to plot a marker and annotate with grid_values
    for i in range(len(y)):
        for j in range(len(x)):
            ax.plot(xx[i, j], yy[i, j], 'ro')
            ax.text(xx[i, j]+offset[0], yy[i, j]+offset[1], str(grid_values[j, i]),
                    fontsize=10, color='black', ha='center', va='center')

    # Set limits and grid details

def plot_chunkify():
    nbatch = 2
    cshape = (2, 2)
    mshape = (2, 3)
    dshape = [cshape[i] * mshape[i] for i in range(2)]

    dsize = [nbatch,] + dshape

    data = np.arange(nbatch)
    data = data.repeat(np.prod(dshape)).reshape(*dsize)
    loc = torch.tensor([[5, 7],
                        [0, 1]])
    block = Block(location=loc, data=torch.tensor(data))

    chunks = chunkify(block, cshape)

    # plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    col_codes = generate_colors(block.size()[0])
    for i in range(block.size()[0]):
        loc = block.location[i]
        c0 = loc.numpy()
        c1 = c0 + np.array(dsize[1:])
        grid_values = block.data[i].numpy()
        plot_grid_on_axis(ax, c0, c1, grid_values, color=col_codes[i], linestyle='-')

    col_codes = generate_colors(chunks.size()[0])
    for i in range(chunks.size()[0]):
        loc = chunks.location[i]
        c0 = loc.numpy()
        c1 = c0 + np.array(cshape)
        grid_values = chunks.data[i].numpy()
        plot_grid_on_axis(ax, c0, c1, grid_values, color=col_codes[i], linestyle='-.')

    # ax.grid(True)
    plt.tight_layout()
    plt.savefig('chunkify_2d.png')
    plt.close()

def test_chunksum():
    nbatch = 2
    cshape = (2, 2)
    mshape = (2, 3)
    dshape = [cshape[i] * mshape[i] for i in range(2)]

    dsize = [nbatch,] + dshape

    data = np.arange(nbatch)
    data = data.repeat(np.prod(dshape)).reshape(*dsize)
    loc = torch.tensor([[5, 7],
                        [0, 1]])
    block = Block(location=loc, data=torch.tensor(data))

    chunksum = ChunkSum(cshape)
    chunks = chunksum(block)

    # hard coded
    expected_loc = torch.tensor([
            [0, 0], # [[0, 1], [0, 1]],
            [0, 2], # [[1, 1], [1, 1]],
            [0, 4], # [[1, 1], [1, 1]],
            [0, 6], # [[1, 0], [1, 0]],
            [2, 0], # [[0, 1], [0, 1]],
            [2, 2], # [[1, 1], [1, 1]],
            [2, 4], # [[1, 1], [1, 1]],
            [2, 6], # [[1, 0], [1, 0]]
        ])
    expected_data = torch.tensor([
            [[0, 1], [0, 1]],
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
            [[1, 0], [1, 0]]
        ])
    assert torch.equal(expected_loc, chunks.location)
    assert torch.equal(expected_data, chunks.data), f'{chunks.data}'


def plot_chunksum():
    nbatch = 2
    cshape = (2, 2)
    mshape = (2, 3)
    dshape = [cshape[i] * mshape[i] for i in range(2)]

    dsize = [nbatch,] + dshape

    data = np.arange(nbatch)
    data = data.repeat(np.prod(dshape)).reshape(*dsize)
    loc = torch.tensor([[5, 7],
                        [0, 1]])
    block = Block(location=loc, data=torch.tensor(data))

    chunksum = ChunkSum(cshape)
    chunks = chunksum(block)

    # plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    col_codes = generate_colors(block.size()[0])
    for i in range(block.size()[0]):
        loc = block.location[i]
        c0 = loc.numpy()
        c1 = c0 + np.array(dsize[1:])
        grid_values = block.data[i].numpy()
        plot_grid_on_axis(ax, c0, c1, grid_values, color=col_codes[i], linestyle='-')

    col_codes = generate_colors(chunks.size()[0])
    for i in range(chunks.size()[0]):
        loc = chunks.location[i]
        c0 = loc.numpy()
        c1 = c0 + np.array(cshape)
        grid_values = chunks.data[i].numpy()
        plot_grid_on_axis(ax, c0, c1, grid_values, offset=(0.3, 0.3), color=col_codes[i], linestyle='-.')

    # ax.grid(True)
    plt.tight_layout()
    plt.savefig('chunksum_2d.png')
    plt.close()

def main():
    test_vdim()
    test_to_tensor()
    test_spoint_and_gpoint()
    test_batched_spoint()
    test_envelope()
    plot_chunkify()
    test_chunksum()
    plot_chunksum()

if __name__ == '__main__':
    main()
