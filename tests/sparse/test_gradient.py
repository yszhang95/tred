import torch
from tred.blocking import Block
from tred.sparse import chunkify, chunkify_bins
from tred.chunking import accumulate

'data0'
"""
0,0,0,0
0,1,0,0
0,2,3,0
0,0,0,0
"""
'data1'
"""
0,0,0,0
0,0,0,1E-3
0,0,4,0
0,0,0,0
"""
'summed'
"""
0,0,0,0
0,1,0,1
0,2,7,0
0,0,0,0
"""

location = torch.tensor([[1,1],[1,2]])
data = torch.tensor([[[2,3], [1,0]],
            [[4,0], [0,1]]], dtype=torch.float32, requires_grad=True)

a0 = torch.tensor([[0,0,0,0], [0,2,3,0], [0,1,0,0], [0,0,0,0]], requires_grad=True, dtype=torch.float32)
a1 = torch.tensor( [ [0,0,0,0], [0,0,4,0], [0,0,0,1], [0,0,0,0], ], requires_grad=True, dtype=torch.float32)
# summed2d = torch.tensor([[0,0,0,0],[0,2,7,0],[0,1,0,1],[0,0,0,0]], dtype=torch.float32, requires_grad=True)

def test_chunkify_bins():
    shape = (2,2)
    b = Block(location=location, data=data)
    b0 = accumulate(chunkify(b, shape))
    b1 = accumulate(chunkify_bins(b, shape))
    assert torch.equal(b0.location, b1.location)
    assert torch.allclose(b0.data, b1.data, atol=1E-5, rtol=1E-5)


def test_gradient():
    b = Block(location=location, data=data)
    #b1 = accumulate(chunkify_bins(b, (4,4)))
    b1 = accumulate(chunkify_bins(b, (4,4)))
    # b1.data.retain_grad()
    s2 = b1.data.square().sum()
    s2.backward()

    suma01 = a0+a1
    summed2 = suma01.square().sum()
    summed2.backward()

    # assert torch.allclose(b1.data.grad, summed2d.grad), "{bg.data.grad}, {summed2d.grad}"
    a0grad_truncation = a0.grad.clone().detach()[location[0,0]:location[0,0]+2, location[0,1]:location[0,1]+2]
    print(a0grad_truncation, b.data.grad[0], a0.grad)
    assert torch.allclose(b.data.grad[0], a0grad_truncation)


if __name__ == '__main__':
    test_chunkify_bins()
    test_gradient()
