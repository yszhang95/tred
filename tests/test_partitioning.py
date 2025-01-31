from tred.partitioning import deinterlace
from tred.blocking import Block
from tred.util import to_tuple
import torch

def test_impacts():
    '''
    Do de-interlacing.
    '''
    # The per-dimension step size of the interlacing.  A step of 1 effectively
    # means the dimension is not interlaced.  This also gives a "super shape" of
    # a conceptual tensor that each element holds the part of a block that was
    # deinterlaced starting at the corresponding impact position.
    step_sizes = torch.tensor([ 1, 10, 10,  1], dtype=torch.int32)

    # Number of steps of the given size that spans the entire dimension.  This
    # is also the shape of the each partitioned block.
    lace_shape = torch.tensor([2,  3,  2, 10], dtype=torch.int32)

    # The total batched tensor shape is then the product of the two.  Note, this
    # test goes directly backward.  A real app must have its full tensor shape
    # pre-defined such that the size of each dimension is an integer multiple of
    # the dimension step size.
    shape = step_sizes * lace_shape

    # The batched location must be on grid points corresponding to super-grid
    # points.
    location = torch.tensor([[-20,-30,-50],
                             [200,300,500]], dtype=torch.int32)

    # The .location for the parts is expected to be on the super-grid.
    super_location = location / step_sizes[1:]

    # the batched, interlaced tensor
    data = torch.arange(torch.prod(shape)).reshape(to_tuple(shape))

    # bundle into a Block
    print(f'{data.shape=}')
    block = Block(location=location, data=data)

    # Perform the de-interlacing.  This yields a generator.  Normal could should
    # not wrap in list() as that will expand memory.  We do it here to perform
    # some extra checks.
    all_imps = list(deinterlace(block, step_sizes[1:]))

    assert len(all_imps) == torch.prod(step_sizes)
                    
    for imp in all_imps:
        assert imp.vdim == 3
        assert imp.nbatches == 2
        assert torch.all(torch.tensor(imp.data.shape) == lace_shape)
        assert torch.all(imp.location == super_location)
