from tred.blocking import Block, apply_slice
from tred.util import to_tensor
import torch

def test_blocking_slice():
    b = Block(data=torch.arange(2*3*4).reshape(2,3,4),
              location=torch.zeros(2*2).reshape(2,2))
    # print(f'{b.location.shape=}\n{b.location}')
    # print(f'{b.data.shape=}\n{b.data}')
    s = apply_slice(b, (slice(1,2), slice(2,3)))
    # print(f'{s.location.shape=}\n{s.location}')
    # print(f'{s.data.shape=}\n{s.data}')
    assert s.location.shape == b.location.shape
    assert torch.all(s.shape == torch.tensor([1,1]))
    assert s.data.shape == torch.Size([2,1,1])

