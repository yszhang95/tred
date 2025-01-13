import torch
from tred.indexing import union_bounds, crop_batched
from tred.util import to_tensor
from functools import reduce
    
def nele(t):
    return reduce(torch.mul, t, 1)

def test_crop_batched():
    bshape = (2,3,4,5)
    bnele = nele(bshape)
    data = torch.arange(bnele).reshape(bshape).to(dtype=torch.int32)
    offsets = torch.tensor([[1, 2, 3],[2, 3, 4]], dtype=torch.int32)
    inner = to_tensor(data.shape[1:])

    outer = torch.tensor([10, 20, 30], dtype=torch.int32)
    inds = crop_batched(offsets, inner, outer)

    assert len(inds) == bnele
    env = torch.zeros([2, 10,20,30], dtype=torch.int32)
    env.flatten()[inds] = data.flatten()


def test_union_bounds():
    block = torch.rand((2,3,4,6))
    print(block.shape)
    offset = torch.tensor([ (-1,-2,-3), (1,2,3) ])
    print(offset)
    pos = union_bounds(block.shape[1:], offset)
    assert len(pos) == 2
    assert all([s>0 for s in pos[1]])
    assert torch.all(pos[0] == offset[0])



#c = z.reshape(-1,2,p//2,2,m//2,2,n//2).permute(1,3,5,0,2,4,6).reshape(-1,p//2,m//2,n//2)
