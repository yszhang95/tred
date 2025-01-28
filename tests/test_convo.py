from tred.convo import dft_shape, zero_pad
from tred.util import to_tensor, to_tuple

import torch

def test_dft_shape():
    ten = torch.arange(2*3*4).reshape(2,3,4)
    ker = torch.arange(1*2*3).reshape(1,2,3)
    assert dft_shape(ten,ker) == (2, 4, 6)

    ten = torch.unsqueeze(ten, 0)
    assert dft_shape(ten,ker) == (2, 4, 6)


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
    

    # inner padding by bracketing zero_pad with roll's.

    tr = torch.roll(t, shifts=to_tuple(h_shape), dims=(0,1,2))
    trp = zero_pad(tr, to_tuple(p_shape))
    trpr = torch.roll(trp, shifts=to_tuple(-h_shape), dims=(0,1,2))

    print(f'{t=}')
    print(f'{tr=}')
    print(f'{trp=}')
    print(f'{trpr=}')

    assert torch.all(trpr[:1,:1,:2] == t[:1, :1, :2])
    assert torch.all(trpr[1:-1, 1:-1, 2:-2] == 0)

    
