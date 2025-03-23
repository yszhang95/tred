#!/usr/bin/env python
'''
Supplemental tests for tests/test_blocking.py
'''

import pytest
import torch
import matplotlib.pyplot as plt
from tred.blocking import Block, apply_slice, batchify
from tred.types import index_dtype


def test_block_init_with_shape():
    device = torch.device("cuda")
    loc = torch.tensor([1, 2], dtype=index_dtype, device=device)
    shape = torch.Size([4, 5])
    block = Block(loc, shape=shape)
    assert torch.equal(block.location, loc.unsqueeze(0))
    assert torch.equal(block.shape, shape), f'block.shape {block.shape}, shape {shape}'
    assert block.nbatches == 1
    assert block.vdim == 2
    assert block.device == loc.device
    assert block.size() == torch.Size([1, 4, 5])

    device = torch.device("cuda")
    loc = torch.tensor([1, 2], dtype=index_dtype, device=device)
    shape = torch.tensor([4, 5], device=device)
    block = Block(loc, shape=shape)
    assert torch.equal(block.shape, shape), f'block.shape {block.shape}, shape {shape}'
    assert block.nbatches == 1
    assert block.vdim == 2
    assert block.device == loc.device
    assert block.size() == torch.Size([1, 4, 5])

    device = torch.device("cpu")
    loc = torch.tensor([1, 2], dtype=index_dtype, device=device)
    shape = torch.tensor([4, 5], device=device)
    block = Block(loc, shape=shape)
    assert torch.equal(block.shape, shape), f'block.shape {block.shape}, shape {shape}'
    assert block.nbatches == 1
    assert block.vdim == 2
    assert block.device == loc.device
    assert block.size() == torch.Size([1, 4, 5])

def test_block_init_with_data():
    loc = torch.tensor([[0, 0], [10, 10]], dtype=index_dtype)
    data = torch.randn(2, 3, 3)
    block = Block(loc, data=data)
    assert torch.equal(block.location, loc)
    assert torch.equal(block.data, data)
    assert block.shape == torch.tensor([3, 3])
    assert block.nbatches == 2
    assert block.vdim == 2

def test_block_set_shape_validation():
    loc = torch.tensor([1, 2], dtype=index_dtype)
    block = Block(loc, shape=torch.tensor([3, 3], dtype=index_dtype))
    with pytest.raises(ValueError):
        block.set_shape(torch.tensor([[3, 3]], dtype=index_dtype))  # batched shape

    with pytest.raises(ValueError):
        block.set_shape(torch.tensor([3], dtype=index_dtype))  # shape length mismatch

def test_block_set_data_validation():
    loc = torch.tensor([[0, 0], [1, 1]], dtype=index_dtype)
    block = Block(loc, shape=torch.tensor([2, 2], dtype=index_dtype))

    with pytest.raises(TypeError):
        block.set_data([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        block.set_data(torch.randn(2, 2, 2, 2))  # too many dims

    with pytest.raises(ValueError):
        block.set_data(torch.randn(3, 2, 2))  # batch mismatch

def test_apply_slice():
    loc = torch.tensor([[0, 0], [10, 10]], dtype=index_dtype)
    data = torch.arange(2*5*5).reshape(2, 5, 5)
    block = Block(loc, data=data)
    sliced = apply_slice(block, (slice(1, 4), slice(2, 5)))

    expected_offset = torch.tensor([1, 2], dtype=index_dtype)
    assert torch.equal(sliced.location, loc + expected_offset)
    assert torch.equal(sliced.data, data[:, 1:4, 2:5])
    assert sliced.shape == torch.tensor([3, 3])

def plot_apply_slice():
    loc = torch.tensor([[0, 0], [10, 10]], dtype=index_dtype)
    data = torch.zeros((2, 5, 5), dtype=torch.int32)
    block = Block(loc, data=data)

    # Apply slice
    sliced = apply_slice(block, (slice(1, 4), slice(2, 5)))

    # Set sliced region to 1
    sliced.data[:] = 1

    # Scatter plot for both blocks
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for i, (title, b) in enumerate(zip(["Block", "Sliced"], [block, sliced])):
        # Create meshgrid of indices
        h, w = b.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        x = x.flatten()
        y = y.flatten()

        # Expand location to match each channel
        global_x = x + b.location[:, 1].unsqueeze(1)
        global_y = y + b.location[:, 0].unsqueeze(1)

        # Values (colors)
        values = b.data.reshape(2, -1)

        for j in range(2):  # for each batch/channel
            axs[i].scatter(global_x[j], global_y[j], s=1+10*values[j], marker='s')

        axs[i].set_title(f'{title}; marker size shows values at grid ponts')
        axs[i].set_aspect('equal')

    axs[1].set_xlim(*axs[0].get_xlim())
    axs[1].set_ylim(*axs[0].get_ylim())

    plt.tight_layout()
    plt.savefig('apply_slice_2d.png')

def test_batchify():
    ten = torch.randn(4, 4)
    batched, added = batchify(ten, ndim=2)
    assert batched.shape == (1, 4, 4)
    assert added

    ten_batched = torch.randn(3, 4, 4)
    batched2, added2 = batchify(ten_batched, ndim=2)
    assert torch.equal(batched2, ten_batched)
    assert not added2

def main():
    test_block_init_with_shape()
    test_block_init_with_data()
    test_block_set_shape_validation()
    test_block_set_data_validation()
    test_apply_slice()
    test_batchify()

    plot_apply_slice()


if __name__ == '__main__':
    main()
