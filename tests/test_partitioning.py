from tred.partitioning import deinterlace
from tred.blocking import Block
from tred.util import to_tuple
import torch

def test_impacts():
    '''
    Do de-interlacing.
    '''

    # assume batched-3D: (batch, pitch0, pitch1, drift)

    # The per-dimension step size of the interlacing.  A step of 1 effectively
    # means the dimension is not interlaced.  This also gives a "super shape" of
    # a conceptual tensor that each element holds the part of a block that was
    # deinterlaced starting at the corresponding impact position.
    step_sizes = torch.tensor([ 1, 10, 10,  1], dtype=torch.int32)

    # Number of steps of the given size that spans the entire dimension.  This
    # is also the shape of the each partitioned block.  In pixel det terms this
    # corresponds to: 2 batches, 3x2 pixels, 10 ticks.
    lace_shape = torch.tensor([2,  3,  2, 10], dtype=torch.int32)

    # The total batched tensor shape is then the product of the two.  Note, this
    # test goes directly backward.  A real app must have its full tensor shape
    # pre-defined such that the size of each dimension is an integer multiple of
    # the dimension step size.
    shape = step_sizes * lace_shape

    # The batched, interlaced tensor filled with bogus data.
    data = torch.arange(torch.prod(shape)).reshape(to_tuple(shape))


    # # The batched location must be on grid points corresponding to super-grid
    # # points.
    # location = torch.tensor([[-20,-30,-50],
    #                          [200,300,500]], dtype=torch.int32)

    # # The .location for the parts is expected to be on the super-grid.
    # super_location = location / step_sizes[1:]

    # # bundle into a Block
    # print(f'{data.shape=}')
    # block = Block(location=location, data=data)

    # Perform the de-interlacing.  This yields a generator.  Normal could should
    # not wrap in list() as that will expand memory.  We do it here to perform
    # some extra checks.
    laces = list(deinterlace(data, step_sizes))

    assert len(laces) == torch.prod(step_sizes)

    for lace in laces:
        assert torch.all(torch.tensor(lace.shape) == lace_shape)

    # not consider batch dim
    for i in range(step_sizes[1]):
        for j in range(step_sizes[2]):
            for k in range(step_sizes[3]):
                pass
                ind = i*step_sizes[2]*step_sizes[3] + j * step_sizes[3] + k
                assert torch.equal(laces[ind], data[:,i::step_sizes[1],
                                                    j::step_sizes[2],k::step_sizes[3]])
