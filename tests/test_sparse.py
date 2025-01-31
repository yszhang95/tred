#!/usr/bin/env pytest
import pytest
import torch
from tred.sparse import SGrid, fill_envelope, reshape_envelope, chunkify
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

def test_sparse_funcs_multichunk():
    sgrid = SGrid([10,10])
    loc = torch.tensor([-1,-1])
    block = Block(loc, data=torch.ones([12,12]))
    #chunks = block_chunk(sgrid, b)
    envelope = sgrid.envelope(block)
    assert torch.all(envelope.location[0] == torch.tensor([-10,-10]))
    fill_envelope(envelope, block)
    assert torch.all(envelope.location[0] == torch.tensor([-10,-10]))

    chunks = reshape_envelope(envelope, sgrid.spacing)

    print(f'{chunks.shape=}')
    for ichunk in range(chunks.nbatches):
        print(f'{ichunk}: loc={chunks.location[ichunk]}: tot={torch.sum(chunks.data[ichunk])}')

def test_sparse_chunkify():
    '''
    Test all in one chunkify() function
    '''
    loc = torch.tensor([-1,-1])
    block = Block(loc, data=torch.ones([12,12]))
    chunk_shape = (10,10)
    chunks = chunkify(block, chunk_shape)
    print(f'{chunks.shape=} {chunk_shape=}')
    for ichunk in range(chunks.nbatches):
        print(f'{ichunk}: loc={chunks.location[ichunk]}: tot={torch.sum(chunks.data[ichunk])}')

