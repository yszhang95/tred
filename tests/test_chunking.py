
from tred.chunking import content as chunk_content, location as chunk_location
from tred.blocking import Block
import torch

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
    loc = torch.zeros([nbatch, len(cshape)])
    print(f'test_chunking_2d: {loc.shape=}')

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

    print(f'{ten.shape=} {loc.shape=}')

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

    rs = chunk_content(Block(location=loc, data=ten), cshape)

    print(f'{rs2.shape=}')
    print(f'{rs.shape=}')

    for b in range(nbatch):

        for x in range(mshape[0]):

            for y in range(mshape[1]):

                val = int(str(b+1)+str(x+1)+str(y+1))

                print(f'{b=} {x=} {y=}\n{rs[b, x, y]}')

                assert torch.all(rs[b, x, y] == val)


def test_chunk_location_3d():
    '''
    Test tred.chunking.location() in 3D
    '''
    cshape = torch.tensor([3,4,5])
    # number of bins per envelope dimension
    mshape = torch.tensor([3,2,1])
    eshape = cshape * mshape

    nbatch = 2
    fullshape = torch.tensor([nbatch] + eshape.tolist())

    offsets = (torch.rand([nbatch, len(cshape)])*10).to(dtype=torch.int)

    locs = chunk_location(Block(location=offsets, shape=eshape), cshape)

    for b in range(nbatch):
        for x in range(mshape[0]):
            for y in range(mshape[1]):
                for z in range(mshape[2]):
                    l = locs[b,x,y,z]
                    o = offsets[b] + torch.tensor(
                       (x*cshape[0],y*cshape[1],z*cshape[2])
                    )
                    assert torch.equal(l, o), f"{l}, {o}"


def test_chunking_3d():
    '''
    Test tred.chunking.content() in 3D

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
    loc = torch.zeros([nbatch, len(cshape)])

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

    print(f'{ten.shape=}')
    rs = chunk_content(Block(location=loc, data=ten), cshape)
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

                    # print(f'{b=} {x=} {y=} {z=}\n{rs[b, x, y, z]}')

                    assert torch.all(rs[b, x, y, z] == val)

    print(f'{rs.shape=}')
    print(rs.flatten(0, 3).shape)

