import pytest
import torch
from tred.sparse import BSB, Block
from tred.util import to_tensor

def tensorify(*arrs, **kwds):
    for a in arrs:
        yield to_tensor(a, **kwds)

def tequal(a, b):
    a,b = tensorify(a,b)
    return torch.all(a == b)


def test_sparse_bsb():
    a = BSB([10,20,30])

    assert tequal(a.spoint([1,1,1]), [0,0,0])
    assert tequal(a.spoint([10,20,30]), [1,1,1])
    assert tequal(a.spoint([-1,-1,-1]), [-1,-1,-1])


    with pytest.raises(ValueError):
        b = Block(torch.arange(2*3*4*5).reshape((2,3,4,5)), [1,2,3])
        bb = a.batchify(b)

    with pytest.raises(ValueError):
        b = Block(torch.arange(3*4*5).reshape((3,4,5)), [[1,2,3]])
        bb = a.batchify(b)

    with pytest.raises(ValueError):
        b = Block(torch.arange(2*3*4*5).reshape((2,3,4,5)), [[1,2,3]])
        bb = a.batchify(b)

    b = Block(torch.arange(2*3*4*5).reshape((2,3,4,5)), [[1,2,3],[2,3,4]])
    print(f'{b.data.shape=}')
    bb = a.batchify(b)
    print(f'{bb.data.shape=}')

    e = a.make_envelope(bb)
    print(f'{e.data.shape=} {e.location.shape=} {e.location} {a.spacing}')

    c = a.make_chunks(e)

def test_sparse_blerg():
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

