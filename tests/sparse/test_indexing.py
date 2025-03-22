#!/usr/bin/env python
'''
Supplemental functions to tests/test_indexing.py
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from tred.indexing import shape_meshgrid, crop, crop_batched, union_bounds

def test_crop():
    offset = torch.tensor((1,2))
    inner = torch.tensor((3,3))
    outer = torch.tensor((10,10))

    parent = torch.zeros(outer.tolist())
    parent2 = parent.clone().detach()

    child = torch.ones(inner.tolist())
    indices = [slice(o, o + i) for o, i in zip(offset, inner)]
    parent[indices] = child

    inds = crop(offset, inner, outer)

    parent2.flatten()[inds] = child.flatten()

def box2d(c0, c1):
    return np.array([[c0[0], c0[0], c1[0], c1[0], c0[0]],
                     [c0[1], c1[1], c1[1], c0[1], c0[1]]])

def plot_crop_batched_2d():
    inner = torch.tensor(((2,2), (3,3)))
    outer = torch.tensor((10, 10))
    offsets = torch.tensor([(3,3), (1,2)])

    grid = shape_meshgrid(outer)

    data = torch.zeros([len(offsets),] + outer.tolist())

    inds = crop_batched(offsets, inner, outer)
    data.flatten()[inds] = 1

    fig, axes = plt.subplots(nrows=len(offsets), ncols=1, figsize=(8, 6*len(offsets)))
    for i in range(len(offsets)):
        axes[i].scatter(grid[0], grid[1], c=data[i])
        axes[i].set_title(label=f'ibatch {i}; offset {offsets[i].tolist()}; shape {inner[i].tolist()}')
    plt.savefig('crop_batched_2d.png')

def plot_union_bounds_2d():
    shape = (2,2)
    offsets = torch.tensor([(3,3), (1,2)])

    pmin, pmax = union_bounds(shape, offsets)
    pmax = pmin + pmax
    boundx, boundy = box2d(pmin.numpy(), pmax.numpy())
    plt.plot(boundx, boundy, 'o-', label='bound')
    for i in range(len(offsets)):
        x, y = box2d(offsets[i].numpy(), (offsets[i] + torch.tensor(shape)).numpy())
        plt.plot(x, y, 'o-', label=f'ibatch{i}')
    plt.legend()
    plt.savefig('union_bounds_2d.png')


def main():
    test_crop()
    plot_union_bounds_2d()
    plot_crop_batched_2d()

if __name__ == '__main__':
    main()
