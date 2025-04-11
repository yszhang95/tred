from tred.partitioning import deinterlace_pairs
from tred.blocking import Block
from tred.util import to_tuple
import torch

import pytest

def test_deinterlace_pairs():
    '''
    Do de-interlacing and we have an axis along which will be returned in pairs.
    '''

    # assume batched-3D: (batch, pitch0, pitch1, drift)
    step_sizes = torch.tensor([ 1, 10, 10,  1], dtype=torch.int32)
    lace_shape = torch.tensor([2,  3,  2, 10], dtype=torch.int32)
    shape = step_sizes * lace_shape

    # The batched, interlaced tensor filled with bogus data.
    # axis = 1
    data = torch.arange(torch.prod(shape)).reshape(to_tuple(shape))
    laces = list(deinterlace_pairs(data, step_sizes, pair_axis=1))

    assert len(laces) == torch.prod(step_sizes) // 2

    for lace in laces:
        assert torch.all(torch.tensor(lace[0].shape) == lace_shape)
        assert torch.all(torch.tensor(lace[1].shape) == lace_shape)

    # not consider batch dim
    # axis = 1
    # we expect order (2,3,1)
    vdim = [step_sizes[2], step_sizes[3], step_sizes[1]//2]
    for j in range(vdim[0]):
        for k in range(vdim[1]):
            for i in range(vdim[2]):
                ind = vdim[1]*vdim[2]*j + vdim[2]*k + i
                assert torch.equal(laces[ind][0], data[:,i::step_sizes[1],
                                                    j::step_sizes[2],k::step_sizes[3]])
                assert torch.equal(laces[ind][1], data[:,step_sizes[1]-1-i::step_sizes[1],
                                                    j::step_sizes[2],k::step_sizes[3]])
    # axis = -3, equivalent to axis = 1
    laces_new = list(deinterlace_pairs(data, step_sizes, pair_axis=-3))
    assert len(laces_new) == len(laces)
    for l1, l2 in zip(laces_new, laces):
        assert torch.equal(l1[0], l2[0])
        assert torch.equal(l1[1], l2[1])

    # axis = 2
    data = torch.arange(torch.prod(shape)).reshape(to_tuple(shape))
    laces = list(deinterlace_pairs(data, step_sizes, pair_axis=2))

    assert len(laces) == torch.prod(step_sizes) // 2

    for lace in laces:
        assert torch.all(torch.tensor(lace[0].shape) == lace_shape)
        assert torch.all(torch.tensor(lace[1].shape) == lace_shape)

    # axis = 2
    # we expect order (1,3,2)
    vdim = [step_sizes[1], step_sizes[3], step_sizes[2]//2]
    for i in range(vdim[0]):
        for k in range(vdim[1]):
            for j in range(vdim[2]):
                ind = vdim[1]*vdim[2]*i + vdim[2]*k + j
                assert torch.equal(laces[ind][0], data[:,i::step_sizes[1],
                                                    j::step_sizes[2],k::step_sizes[3]])
                assert torch.equal(laces[ind][1], data[:,i::step_sizes[1],
                                                    step_sizes[2]-1-j::step_sizes[2],k::step_sizes[3]])

    # simple test using input with reflective symmetry
    data = torch.tensor([[0,1,2,2,1,0], [1,2,3,3,2,1]])
    laces = list(deinterlace_pairs(data, torch.tensor((1,2)), pair_axis=1))
    assert len(laces) == 1
    for lace in laces:
        assert lace[0].shape == torch.Size([2,3])
        assert lace[1].shape == torch.Size([2,3])
    assert torch.equal(laces[0][0], torch.tensor([[0,2,1], [1,3,2]]))
    assert torch.equal(laces[0][1], torch.tensor([[1,2,0], [2,3,1]]))
    assert torch.equal(laces[0][0], torch.flip(laces[0][1], dims=(1,)))

def main():
    test_deinterlace_pairs()

if __name__ == '__main__':
    main()
