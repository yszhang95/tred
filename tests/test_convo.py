from tred.convo import dft_shape, zero_pad
from tred.util import to_tensor, to_tuple

import torch

def test_dft_shape():
    tshape = (2,3,4)
    kshape = (1,2,3)
    cshape = (2, 4, 6)
    assert dft_shape(tshape, kshape) == cshape

    tshape = (42, 2,3,4)        # batched
    assert dft_shape(tshape, kshape) == cshape

def test_zero_pad():
    numel = 2*3*4
    o_shape = to_tensor((2,3,4), dtype=torch.int32)
    h_shape = o_shape // 2
    p_shape = o_shape*2

    t = torch.arange(numel).reshape(*to_tuple(o_shape))
    tp = zero_pad(t, to_tuple(p_shape))

    assert torch.all(to_tensor(tp.shape) == p_shape)
    assert torch.all(tp[2:,:,:] == 0)
    assert torch.all(tp[:,3:,:] == 0)
    assert torch.all(tp[:,:,4:] == 0)
    assert torch.equal(tp[:2,:3,:4], t)


    # inner padding by bracketing zero_pad with roll's.

    tr = torch.roll(t, shifts=to_tuple(h_shape), dims=(0,1,2))
    trp = zero_pad(tr, to_tuple(p_shape))
    trpr = torch.roll(trp, shifts=to_tuple(-h_shape), dims=(0,1,2))

    print(f'{t=}')
    print(f'{tr=}')
    print(f'{trp=}')
    print(f'{trpr=}')

    assert torch.all(trpr[:1,:2,:2] == t[:1, :2, :2]) # from 0 to o_shape - h_shape - 1; first o_shape - h_shape
    assert torch.all(trpr[-1:,-1:,-2:] == t[-1:, -1:, -2:]) # last h_shape
    assert torch.all(trpr[1:-1, 1:-1, 2:-2] == 0) # The joint is of course zeros
    assert torch.all(trpr[1:-1, :, :] == 0) # We actually want unions;
    assert torch.all(trpr[:, 2:-1, :] == 0) # We actually want unions;
    assert torch.all(trpr[:, :, 2:-2] == 0) # We actually want unions;
