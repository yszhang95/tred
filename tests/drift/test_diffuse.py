import torch
import pytest

from tred.drift import diffuse

# Assume the diffuse function is already defined

def test_diffuse_scalar_diffusion_no_sigma():
    # diffusion provided as a scalar (not a tensor)
    dt = torch.tensor([1.0, 2.0, 3.0])
    diffusion = 5.0
    # Expected: sigma = sqrt(2 * diffusion * dt)
    expected = torch.sqrt(2 * 5.0 * dt)
    result = diffuse(dt, diffusion)
    # Because diffusion was scalar, the function sets squeeze=True so that
    # the extra dimension is removed.
    assert torch.allclose(result, expected)

def test_diffuse_vector_diffusion_no_sigma():
    # diffusion provided as a 1D tensor (vector)
    dt = torch.tensor([1.0, 2.0])
    diffusion = torch.tensor([4.0, 5.0, 6.0])
    # Expected: sigma = sqrt(2 * dt[:, None] * diffusion[None, :])
    expected = torch.sqrt(2 * dt[:, None] * diffusion[None, :])
    result = diffuse(dt, diffusion)
    # In this case diffusion is already a tensor, so squeeze remains False and
    # the output shape is (npts, diffusion_length)
    assert torch.allclose(result, expected)

def test_diffuse_with_sigma():
    # Test the case where an initial sigma is provided.
    dt = torch.tensor([1.0, 2.0])
    diffusion = 1.0  # scalar diffusion
    sigma_initial = torch.tensor([0.5, 1.0])  # one sigma per time point (1D)
    # Expected: sigma = sqrt(2*diffusion*dt + sigma_initial^2)
    expected = torch.sqrt(2 * 1.0 * dt + sigma_initial**2)
    result = diffuse(dt, diffusion, sigma_initial)
    # For scalar diffusion, squeeze=True so that the result is squeezed to shape (npts,)
    assert result.shape == sigma_initial.shape
    assert torch.allclose(result, expected)

def test_diffuse_negative_dt():
    # Test with a negative dt causing the argument of sqrt to be negative.
    dt = torch.tensor([-1.0])
    diffusion = 1.0
    result = diffuse(dt, diffusion)
    # sqrt(2*1*(-1)) = sqrt(-2) returns nan, which the function then sets to 0.
    expected = torch.tensor(0.0)
    assert torch.allclose(result, expected)

def test_dt_shape_exception():
    # dt must be 1D. Here we pass a 2D tensor.
    dt = torch.tensor([[1.0, 2.0]])
    diffusion = 1.0
    with pytest.raises(ValueError):
        diffuse(dt, diffusion)

def test_diffusion_shape_exception():
    # diffusion must be 1D. Here we pass a 2D tensor.
    dt = torch.tensor([1.0])
    diffusion = torch.tensor([[1.0]])
    with pytest.raises(ValueError):
        diffuse(dt, diffusion)

def test_sigma_shape_exception():
    # sigma's shape should span the same number of dimensions as diffusion (which is 1 for 1D diffusion)
    # Here we provide a sigma with an extra dimension (2D instead of 1D)
    dt = torch.tensor([1.0, 2.0])
    diffusion = 1.0  # so diffusion is internally converted to a 1D tensor of length 1
    sigma_wrong = torch.tensor([[0.5, 1.0]])  # shape (1, 2) does not match the expected 1D shape for sigma
    with pytest.raises(ValueError):
        diffuse(dt, diffusion, sigma_wrong)
