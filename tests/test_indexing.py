import torch
from tred.indexing import chunk, chunk_location, union_bounds, crop_batched
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


def test_chunking_2d():
    '''
    Explicit 2D chunking alg test.

    See test_chunking_3d below.

    Input shape is (nb, ex, ey) output is (nb, mx, my, cx, cy).
    '''
    cshape = torch.tensor([3,4])
    # number of bins per envelope dimension
    mshape = torch.tensor([3,2])
    eshape = cshape * mshape
    nbatch = 2
    fullshape = torch.tensor([nbatch] + eshape.tolist())

    ten = torch.zeros(fullshape.tolist(), dtype=torch.int32)

    # encode a "chunk ID" common to all elements in a chunk to make sure we get
    # back what we expect
    for b in range(nbatch):

        for x in range(mshape[0]):
            ixb = x*cshape[0]
            ixe = ixb + cshape[0]

            for y in range(mshape[1]):
                iyb = y*cshape[1]
                iye = iyb + cshape[1]

                ten[b, ixb:ixe, iyb:iye] = int(str(b+1)+str(x+1)+str(y+1))

    print(f'{ten.shape=}\n{ten}')

    print(f'{cshape=} {mshape=} {eshape=}')
    rargs1=(-1, mshape[0].item(), cshape[0].item(), mshape[1].item(), cshape[1].item())
    pargs=(0,1,3,2,4)
    rargs2=(-1, mshape[0].item(), mshape[1].item(), cshape[0].item(), cshape[1].item())
    print(f'{rargs1=} {pargs=} {rargs2=} (want)')

    rs2 = ten.reshape(*rargs1).permute(*pargs).reshape(*rargs2)
    # rs2 = ten.reshape(-1,
    #                   mshape[0], cshape[0], # 3, 2
    #                   mshape[1], cshape[1])\
    #                  .permute(0,1,3,2,4)\
    #                  .reshape(-1,
    #                           mshape[0], mshape[1],
    #                           cshape[0], cshape[1])

    rs = chunk(ten, cshape)

    print(f'{rs2.shape=}\n{rs2}')
    print(f'{rs.shape=}\n{rs}')

    for b in range(nbatch):

        for x in range(mshape[0]):

            for y in range(mshape[1]):

                val = int(str(b+1)+str(x+1)+str(y+1))

                print(f'{b=} {x=} {y=}\n{rs[b, x, y]}')

                assert torch.all(rs[b, x, y] == val)


def test_chunk_locating_3d():
    '''
    Test tred.indexing.chunk_location() in 3D
    '''
    cshape = torch.tensor([3,4,5])
    # number of bins per envelope dimension
    mshape = torch.tensor([3,2,1])
    eshape = cshape * mshape

    nbatch = 2
    fullshape = torch.tensor([nbatch] + eshape.tolist())

    offsets = (torch.rand(fullshape.tolist())*10).to(dtype=torch.int)

    loc = chunk_location(offsets, eshape, cshape)
    

def test_chunking_3d():
    '''
    Test tred.indexing.chunk() in 3D

    This test constructs "envelopes" with elements that encode their chunk
    index, applies chunking and then requires each chunk to have elements all
    the same and correct value.
    '''

    cshape = torch.tensor([3,4,5])
    # number of bins per envelope dimension
    mshape = torch.tensor([3,2,1])
    eshape = cshape * mshape

    nbatch = 2
    fullshape = torch.tensor([nbatch] + eshape.tolist())
    print(f'{fullshape=}')

    ten = torch.zeros(fullshape.tolist(), dtype=torch.int32)

    for b in range(nbatch):

        for x in range(mshape[0]):
            ixb = x*cshape[0]
            ixe = ixb + cshape[0]

            for y in range(mshape[1]):
                iyb = y*cshape[1]
                iye = iyb + cshape[1]

                for z in range(mshape[2]):
                    izb = z*cshape[2]
                    ize = izb + cshape[2]

                    ten[b, ixb:ixe, iyb:iye, izb:ize] = int(str(b+1)+str(x+1)+str(y+1)+str(z+1))

    print(f'{ten.shape=}\n{ten=}')
    rs = chunk(ten, cshape)
    rs2 = ten.reshape(-1,
                     mshape[0], cshape[0],
                     mshape[1], cshape[1],
                     mshape[2], cshape[2])\
                     .permute(0,1,3,5,2,4,6)\
                     .reshape(-1,
                              mshape[0], mshape[1], mshape[2],
                              cshape[0], cshape[1], cshape[2])

    for b in range(nbatch):

        for x in range(mshape[0]):

            for y in range(mshape[1]):

                for z in range(mshape[2]):
                    val = int(str(b+1)+str(x+1)+str(y+1)+str(z+1))

                    print(f'{b=} {x=} {y=} {z=}\n{rs[b, x, y, z]}')

                    assert torch.all(rs[b, x, y, z] == val)

    print(f'{rs.shape=}')
    print(rs.flatten(0, 3).shape)



#c = z.reshape(-1,2,p//2,2,m//2,2,n//2).permute(1,3,5,0,2,4,6).reshape(-1,p//2,m//2,n//2)
