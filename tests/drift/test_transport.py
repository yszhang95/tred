#!/usr/bin/env python

import torch

from tred.drift import transport

def test_transport():
    # Test 1: Basic positive velocity
    locs = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor(4.0)
    velocity = torch.tensor(2.0)
    expected = torch.tensor([2.0, 1.5, 1.0])
    assert torch.allclose(transport(locs, target, velocity), expected), "Test 1 failed"

    # Test 2: Basic negative velocity
    locs = torch.tensor([5.0, 3.0, 1.0])
    target = torch.tensor(-1.0)
    velocity = torch.tensor(-2.0)
    expected = torch.tensor([3.0, 2.0, 1.0])
    assert torch.allclose(transport(locs, target, velocity), expected), "Test 2 failed"

    # Test 3: Zero velocity (Should handle inf or error)
    locs = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor(5.0)
    velocity = torch.tensor(0.0)
    try:
        result = transport(locs, target, velocity)
        assert torch.isinf(result).all(), "Test 3 failed: Expected infinity"
    except ZeroDivisionError:
        pass  # Also acceptable if an exception is raised

    # Test 4: Target same as locations (Zero time expected)
    locs = torch.tensor([3.0, 3.0, 3.0])
    target = torch.tensor(3.0)
    velocity = torch.tensor(5.0)
    expected = torch.tensor([0.0, 0.0, 0.0])
    assert torch.allclose(transport(locs, target, velocity), expected), "Test 4 failed"

    # Test 5: Fractional results
    locs = torch.tensor([1.5, 2.5, 3.5])
    target = torch.tensor(5.0)
    velocity = torch.tensor(1.5)
    expected = torch.tensor([2.3333, 1.6667, 1.0])
    assert torch.allclose(transport(locs, target, velocity), expected, atol=1e-4), "Test 5 failed"

    # Test 6: Negative locations
    locs = torch.tensor([-2.0, -1.0, 0.0])
    target = torch.tensor(3.0)
    velocity = torch.tensor(2.0)
    expected = torch.tensor([2.5, 2.0, 1.5])
    assert torch.allclose(transport(locs, target, velocity), expected), "Test 6 failed"

    # Test 7: Broadcasting support
    locs = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor(4.0)  # Scalar as tensor
    velocity = torch.tensor(2.0)  # Scalar as tensor
    expected = torch.tensor([2.0, 1.5, 1.0])
    assert torch.allclose(transport(locs, target, velocity), expected), "Test 7 failed"

    print("All tests passed!")

test_transport()
