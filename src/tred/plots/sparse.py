import torch
import tred.sparse
from tred.indexing import union_bounds


# Must run in jupyter notebook
# $ uv run --with jupyter,k3d  jupyter notebook
import k3d

def bo_volumes(block, offset):
    ret = list()
    for ind, (b,o) in enumerate(zip(block, offset)):
        print(f'{o=} {b.shape=} {torch.sum(b)}')
        bs = torch.tensor(b.shape)
        bounds = torch.vstack((o, o+bs)).T.flatten()
        v = k3d.volume(b, bounds=bounds, color_range=[-1, +1])
        ret.append(v)
    return ret

def k3d_block_offset():
    '''
    Simple test of k3d with overlapping volumes.

    Note, the rendering is not proper and the part of the "front" block that is
    inside the "back" block will actually obscure the "back" block at some
    angles.
    '''


    block = torch.rand((2,3,4,6))
    block[0] -= 1

    offset = torch.tensor([ (-1,-2,-3), (1,2,3) ])
    print(f'{block.shape=} {offset.shape=} {offset}')

    plot = k3d.plot()

    for v in bo_volumes(block, offset):
        plot += v

    # for ind, (b,o) in enumerate(zip(block, offset)):
    #     print(f'{o.shape=} {b.shape=} {torch.sum(b)}')
    #     bs = torch.tensor(b.shape)
    #     bounds = torch.vstack((o, o+bs)).T.flatten()
    #     v = k3d.volume(b, bounds=bounds, color_range=[-1, +1])
    #     plot += v
    # plot.display()


    binsize = torch.tensor([10,20,30])
    a = tred.sparse.BSB(binsize)    
    eblock, eoffset = a.fill(block, offset)
    print(f'{eblock.shape=} {eoffset=}')
    
    for v in bo_volumes(eblock, eoffset):
        plot += v

    plot.display()



def k3d_voxels():
    '''
    Draw blocks as voxels
    '''

    bshape = (3, 4, 6)
    blocks = torch.ones((2,3,4,6), dtype=torch.int)
    blocks[1] += 1
    boffsets = torch.tensor([ (-1,-2,-3), (1,2,3) ], dtype=torch.int)
    moffset, mshape = union_bounds(bshape, boffsets)
    print(f'{moffset=} {mshape=}')
    voxels = torch.zeros(tuple(mshape), dtype=torch.uint8)
    for ind, o in enumerate(boffsets):
        print(f'{o=}')
        o = o - moffset
        s = [slice(start.item(), (start+p).item()) for start,p in zip(o, bshape)]
        print(f'{s=}')
        voxels[*s] = blocks[ind]

    plt_voxels = k3d.voxels(voxels, outlines=True, opacity=0.1)
    plot = k3d.plot()
    plot += plt_voxels
    plot.display()
