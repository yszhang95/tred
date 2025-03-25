import torch
import pytest
from tred.sparse import SGrid  # Replace with the actual module name where SGrid is defined
from tred.blocking import Block  # Replace with the actual module path

def test_vdim():
    # Test that the number of spatial dimensions equals the length of the spacing vector.
    spacing = torch.tensor([2, 3, 4])
    grid = SGrid(spacing)
    assert grid.vdim == 3

def test_to_tensor():
    # Test that to_tensor converts a list to a torch.Tensor on the same device as spacing.
    spacing = torch.tensor([2, 2])
    grid = SGrid(spacing)
    data = [1, 2, 3]
    tensor_data = grid.to_tensor(data)
    assert isinstance(tensor_data, torch.Tensor)
    assert tensor_data.device == spacing.device

def test_spoint_and_gpoint():
    # For spacing [2,2], a grid point [3,4] should map to super-grid index [1,2].
    spacing = torch.tensor([2, 2])
    grid = SGrid(spacing)

    # Non-batched input without offset.
    gpt = torch.tensor([3, 4])
    sp = grid.spoint(gpt)
    expected_sp = torch.tensor([1, 2])
    assert torch.equal(sp, expected_sp)

    # gpoint should convert the super point back to grid coordinates (multiplied by spacing).
    gpt_from_sp = grid.gpoint(sp)
    expected_gpt_from_sp = torch.tensor([2, 4])
    assert torch.equal(gpt_from_sp, expected_gpt_from_sp)

    # Test with offset provided.
    offset = torch.tensor([1, 1])
    sp_offset = grid.spoint(gpt, goffset=offset)
    # Calculation: ([3,4] - [1,1]) // [2,2] = [2,3] // [2,2] = [1,1]
    expected_sp_offset = torch.tensor([1, 1])
    assert torch.equal(sp_offset, expected_sp_offset)

def test_batched_spoint():
    # Test that batched input (2D tensor) for spoint returns the correct batched result.
    spacing = torch.tensor([2, 2])
    grid = SGrid(spacing)

    # Batched grid points.
    gpts = torch.tensor([[3, 4],
                           [5, 6]])
    sp = grid.spoint(gpts)
    # Expected:
    #   Row 1: [3//2, 4//2] = [1, 2]
    #   Row 2: [5//2, 6//2] = [2, 3]
    expected = torch.tensor([[1, 2],
                             [2, 3]])
    assert torch.equal(sp, expected)

def test_envelope():
    # For a given Block (pbounds) with location [1,1] and shape [3,4]
    # and spacing [2,2], the envelope should cover:
    #   minpts = gpoint(spoint([1,1])) = [0,0]    (since 1//2 = 0)
    #   maxpts = gpoint(spoint([3,4]) + 1)
    #          = gpoint(([3//2,4//2] + 1)) = gpoint([1+1,2+1]) = gpoint([2,3])
    #          = [2*2, 3*2] = [4,6]
    # So the envelope should have location [0,0] and shape [4,6].
    spacing = torch.tensor([2, 2])
    grid = SGrid(spacing)

    # Create a Block with location and shape.
    pbounds = Block(location=torch.tensor([[1, 1], [2, 3]]), shape=torch.tensor([3, 4]))
    env = grid.envelope(pbounds)

    expected_location = torch.tensor([[0, 0], [2, 2]])  # Block stores location as a batched tensor.
    expected_shape = torch.tensor([4, 6])
    assert torch.equal(env.location, expected_location), f'envelope location {env.location}, expected_location {expected_location}'
    assert torch.equal(env.shape, expected_shape), f'envelope shape {env.shape}, expected_shape {expected_shape}'
