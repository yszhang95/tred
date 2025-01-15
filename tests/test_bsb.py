#!/usr/bin/env pytest

from tred.bsb import BSB
from tred.blocking import Block
from tred.util import to_tensor

import torch

def test_bsb():
    bsb = BSB([10,10])
    loc = torch.tensor([-1,-1])
    block = Block(loc, data=torch.ones([12,12]))
    bsb.fill(block)

    assert bsb.chunks.nbatches == 9
    assert bsb.cindex.nbatches == 1

    # the location, on the super-grid, of the spacial index
    corigin = bsb.cindex.location[0]
    print(f'{corigin=}')
    assert torch.all(corigin == torch.tensor([-1,-1]))
    # find the bins for all chunks
    bins = (bsb.sgrid.spoint(bsb.chunks.location) - corigin).T
    print(f'{bins=}')
    assert torch.all(bins >= 0)
    assert torch.all(bins < 3)
    
    cindex = bsb.cindex.data[0]
    print(f'{cindex.shape=}')
    assert torch.all(to_tensor(cindex.shape) == torch.tensor([3,3]))

    b_indices = cindex[*bins]
    print(f'{b_indices=}')
    assert torch.all(b_indices == torch.arange(bsb.chunks.nbatches))

