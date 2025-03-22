#!/usr/bin/env python
'''
Supplemental functions to tests/test_indexing.py
'''

import torch

from tred.indexing import crop

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
