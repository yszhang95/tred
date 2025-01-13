#!/usr/bin/env pytest
import pytest
import torch
from tred.sparse import BSB, SGrid, fill_envelope, reshape_envelope
from tred.util import to_tensor
from tred.blocking import Block

def tensorify(*arrs, **kwds):
    for a in arrs:
        yield to_tensor(a, **kwds)

def tequal(a, b):
    a,b = tensorify(a,b)
    return torch.all(a == b)


def test_sparse_bsb_guts():
    sgrid = SGrid([10,20,30])

    assert tequal(sgrid.spoint([1,1,1]), [0,0,0])
    assert tequal(sgrid.spoint([10,20,30]), [1,1,1])
    assert tequal(sgrid.spoint([-1,-1,-1]), [-1,-1,-1])


    with pytest.raises(ValueError):
        # Data is batched, location is not.
        b = Block(data=torch.arange(2*3*4*5).reshape((2,3,4,5)), location=[1,2,3])

    # Location is batched, data is not.  Data will be unsqueezed.
    b = Block(data=torch.arange(3*4*5).reshape((3,4,5)), location=[[1,2,3]])

    with pytest.raises(ValueError):
        # Both batched but location is wrong size
        b = Block(data=torch.arange(2*3*4*5).reshape((2,3,4,5)), location=[[1,2,3]])

    b = Block(data=torch.arange(2*3*4*5).reshape((2,3,4,5)), location=[[1,2,3],[2,3,4]])
    assert b.shape is not None

    # Here are the main ingredients of BSB.fill()

    e = sgrid.envelope(b)
    assert e is not None
    assert not hasattr(e, "data")
    print(f'{e.location.shape=} {e.shape=} {sgrid.spacing}')


    fill_envelope(e, b)

    c = reshape_envelope(e, sgrid.spacing)
    assert c is not None
    assert c.data is not None

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
    b = Block(data=block, location=offset)
    a.fill(b)

