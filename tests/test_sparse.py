import torch
from tred.sparse import BSB

def tensorify(*arrs):
    for a in arrs:
        if isinstance(a, torch.Tensor):
            yield a
        else:
            yield torch.tensor(a)

def tequal(a, b):
    a,b = tensorify(a,b)
    return torch.all(a == b)

def test_sparse_bsb_indexing():
    '''
    Test the BSB sparse array-like object
    '''

    nbatch = 2
    block = torch.rand((nbatch,3,4,6))
    offset = torch.tensor([ (-1,-2,-3), (1,2,3) ])
    assert offset.shape[0] == nbatch
    binsize = torch.tensor([10,20,30])
    a = BSB(binsize)    
    eblock, eoffset = a.fill(block, offset)
    print(f'{offset=}\n{eoffset=}')
    assert len(offset.shape) == len(eoffset.shape)
    assert len(block.shape) == len(eblock.shape)
    print(f'{binsize=}')
    for ind in range(nbatch):
        b,o,eb,eo = block[ind], offset[ind], eblock[ind], eoffset[ind]
        bs = torch.tensor(b.shape)
        ebs = torch.tensor(eb.shape)
        print(f'{o=} {o+bs=} --> {eo=} {eo+ebs=}')

