#!/usr/bin/env pytest
import pytest
import torch
from tred.sparse import SGrid, fill_envelope, reshape_envelope, chunkify, index_chunks
from tred.util import to_tensor
from tred.blocking import Block

def compare_values_2d(nbatch_orig, mshape, cshape, cdata, bdata, boff):
    bshape = bdata.shape[1:]
    clist = []
    blist = []
    for ib in range(nbatch_orig):
        for imx in range(mshape[0]):
            for imy in range(mshape[1]):
                for ix in range(cshape[0]):
                    for iy in range(cshape[1]):
                        iox = imx * cshape[0] + ix
                        ioy = imy * cshape[1] + iy
                        ic = imx * mshape[1] + imy 
                        if (
                                iox in range(boff[ib,0], boff[ib,0]+bshape[0])
                                and
                                ioy in range(boff[ib,1], boff[ib,1]+bshape[1])
                        ):
                            clist.append(cdata[ib,ic,ix,iy].item())
                            blist.append(bdata[ib, iox-boff[ib,0], ioy-boff[ib,1]].item())
                        else:
                            clist.append(cdata[ib,ic,ix,iy].item())
                            blist.append(0)
    return torch.tensor(clist), torch.tensor(blist)


def compare_values_3d(nbatch_orig, mshape, cshape, cdata, bdata, boff):
    bshape = bdata.shape[1:]
    clist = []
    blist = []
    for ib in range(nbatch_orig):
        for imx in range(mshape[0]):
            for imy in range(mshape[1]):
                for imz in range(mshape[2]):
                    for ix in range(cshape[0]):
                        for iy in range(cshape[1]):
                            for iz in range(cshape[2]):
                                iox = imx * cshape[0] + ix
                                ioy = imy * cshape[1] + iy
                                ioz = imz * cshape[2] + iz
                                ic = imx * mshape[1] * mshape[2] + imy * mshape[2] + imz
                                if (
                                        iox in range(boff[ib,0], boff[ib,0]+bshape[0])
                                        and
                                        ioy in range(boff[ib,1], boff[ib,1]+bshape[1])
                                        and
                                        ioz in range(boff[ib,2], boff[ib,2]+bshape[2])
                                ):
                                    clist.append(cdata[ib,ic,ix,iy,iz].item())
                                    blist.append(bdata[ib, iox-boff[ib,0], ioy-boff[ib,1], ioz-boff[ib,2]].item())
                                else:
                                    clist.append(cdata[ib,ic,ix,iy,iz].item())
                                    blist.append(0)
    return torch.tensor(clist), torch.tensor(blist)


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

    assert e.location.shape == torch.Size([2,3])
    assert torch.equal(e.shape, torch.tensor([10,20,30]))

    fill_envelope(e, b)

    c = reshape_envelope(e, sgrid.spacing)
    assert c is not None
    assert c.data is not None

    bsize = b.size()

    cdata = c.data.reshape(bsize[0], -1, *sgrid.spacing)
    nchunks = cdata.size(1)
    cshape = cdata.size()[2:]

    esize = e.shape
    mshape = [esize[i].item()//sgrid.spacing[i].item() for i in range(sgrid.vdim)]

    boff = b.location - e.location
    c0, d0 = compare_values_3d(bsize[0], mshape, cshape, cdata, b.data, boff)
    assert torch.equal(c0, d0)


def test_sparse_funcs_multichunk():
    sgrid = SGrid([10,10])
    loc = torch.tensor([-1,-1])
    block = Block(loc, data=torch.ones([12,12]))
    envelope = sgrid.envelope(block)
    assert torch.all(envelope.location[0] == torch.tensor([-10,-10]))
    fill_envelope(envelope, block)
    assert torch.all(envelope.location[0] == torch.tensor([-10,-10]))

    chunks = reshape_envelope(envelope, sgrid.spacing)

    print(f'{chunks.shape=}')
    for ichunk in range(chunks.nbatches):
        print(f'{ichunk}: loc={chunks.location[ichunk]}: tot={torch.sum(chunks.data[ichunk])}')

    bsize = block.size()
    cshape = sgrid.spacing.tolist()
    cdata = chunks.data.reshape(bsize[0], -1, *sgrid.spacing)
    bdata = block.data
    boff = block.location - envelope.location
    mshape = [ envelope.shape[i].item()//cshape[i] for i in range(len(cshape))]
    c0, d0 = compare_values_2d(bsize[0], mshape, cshape, cdata, bdata, boff)
    assert torch.equal(c0, d0), f'{c0}, {d0}'


def test_sparse2d_chunkify():
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

    bsize = block.size()
    cshape = chunk_shape
    cdata = chunks.data.reshape(bsize[0], -1, *cshape)
    bdata = block.data
    boff = block.location - torch.tensor([[-10,-10]]) # I know it is here
    eshape = (30, 30) # I know it is the value.
    mshape = [ eshape[i]//cshape[i] for i in range(len(cshape))]
    c0, d0 = compare_values_2d(bsize[0], mshape, cshape, cdata, bdata, boff)
    assert torch.equal(c0, d0), f'{c0}, {d0}'


def test_sparse_chunkify():
    '''
    Test all in one chunkify() function
    '''
    loc = torch.tensor([-1,-1])
    block = Block(loc, data=torch.ones([12,5]))
    chunk_shape = (10,10)
    chunks = chunkify(block, chunk_shape)
    # block boundaries is (-1, -1) to (11, 4)
    # I expect chunks at (-10, -10),  (-10, 0), (0, -10), (0, 0), (10, -10), (10, 0)
    clocs = torch.tensor([
        (-10, -10),  (-10, 0), (0, -10), (0, 0), (10, -10), (10, 0)
    ])

    b = block
    sgrid = SGrid(chunk_shape)
    e = sgrid.envelope(b)
    fill_envelope(e, b)

    chunk_inds = index_chunks(SGrid(chunk_shape), chunks)

    # only one location is relevant
    torch.equal(chunk_inds.location[0], torch.tensor([-10, -10]))

    positions = torch.zeros_like(clocs)

    for ib in range(chunks.nbatches):
        idx = torch.where(chunk_inds.data[0] == ib)
        positions[ib,0] = idx[0]
        positions[ib,1] = idx[1]
    positions = positions * torch.tensor(chunk_shape) + torch.tensor([[-10,-10]])
    args = torch.argsort(positions[:,0]*2000+positions[:,1])
    assert torch.equal(positions[args], clocs)


    # from docstring
    chunk = chunks
    ic = index_chunks(sgrid, chunk)
    ic_origin = ic.location[0]
    ic_index = ic.data[0]
    s_index = sgrid.spoint(chunk.location, sgrid.gpoint(ic_origin)).T

    # s_index = [s for s in s_index]
    # b_index = ic_index[s_index]
    b_index = ic_index[s_index.tolist()]
    assert torch.all(b_index == torch.arange(chunk.nbatches)), f'{b_index}'

