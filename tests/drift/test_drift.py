import torch
import pytest
from tred.drift import drift

def test_drift_input_validation():
    """Ensure drift function properly handles incorrect input types and shapes."""
    locs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    velocity = 1.0
    diffusion = torch.tensor([0.1, 0.2])
    lifetime = 10.0

    with pytest.raises(ValueError):
        drift(locs, velocity, torch.tensor([[0.1]]), lifetime)  # Incorrect diffusion shape

    with pytest.raises(ValueError):
        drift(locs, velocity, diffusion, lifetime, vaxis=3)  # Out of bounds vaxis

    with pytest.raises(ValueError):
        drift(locs, velocity, diffusion, lifetime, sigma=torch.tensor([[0.1]]))  # Mismatched sigma shape

    with pytest.raises(AttributeError):
        drift(locs, velocity, diffusion.tolist(), lifetime, sigma=torch.tensor([[0.1]]))  # Incorrect diffusion

@pytest.mark.parametrize("velocity,lifetime,target,vaxis", [
    (1.0, 10.0, 5.0, 1),
    (torch.tensor(1.0), torch.tensor(10.0), torch.tensor(5.0), torch.tensor(1)),
])
def test_drift_locs_update(velocity,lifetime,target,vaxis):
    """Check if locs are updated correctly along vaxis."""
    locs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    diffusion = torch.tensor([0.1, 0.2])

    locs_out, _, _, _ = drift(locs, velocity, diffusion, lifetime, target=target, vaxis=vaxis)
    assert torch.allclose(locs_out[:, vaxis], torch.full((len(locs),), target)), "locs[vaxis] should be set to target"
    assert torch.allclose(locs_out[:, 0], locs[:, 0]), "Other loc dimensions should remain unchanged"

@pytest.mark.parametrize("velocity,lifetime,target,vaxis", [
    (2.0, 10.0, 5.0, 0),
    (torch.tensor(2.0), torch.tensor(10.0), torch.tensor(5.0), torch.tensor(0)),
])
def test_drift_times_update(velocity,lifetime,target,vaxis):
    """Check if times are updated correctly."""
    locs = torch.tensor([1.0, 3.0])
    diffusion = 0.1
    lifetime = 10.0
    target = 5.0
    times = torch.tensor([0.5, 1.0])

    dt = (target - locs) / velocity
    _, times_out, _, _ = drift(locs, velocity, diffusion, lifetime, target=target, times=times)
    assert torch.allclose(times_out, times + dt), "Times should be incremented by dt"

@pytest.mark.parametrize("velocity,lifetime,target,vaxis", [
    (-2.0, 10.0, 5.0, 0),
    (torch.tensor(-2.0), torch.tensor(10.0), torch.tensor(5.0), torch.tensor(0)),
])
def test_neg_drift_velocity(velocity,lifetime,target,vaxis):
    '''Check if sigmas are 0, dt < 0 when velocity is negative and target - loc > 0'''
    locs = torch.tensor([1.0, 3.0])
    diffusion = 0.1
    _, times_out, sigma_out, _ = drift(locs, velocity, diffusion, lifetime, target=target)
    assert torch.all(times_out<0), f'Output times, i.e., dt, {times_out} should be negative.'
    assert torch.allclose(sigma_out, torch.tensor(0.), atol=1E-6), f'Output sigma, {sigma_out}, should be 0 when dt is negative.'

def test_default_charge_dtype():
    """Check whether default charge's dtype and its value equal to 1000"""
    locs = torch.tensor([3.0, 3.0])
    velocity = 2.0
    diffusion = 0.1
    lifetime = 10.0 # dummy values
    target = 3.0
    _, _, _, charges_out = drift(locs, velocity, diffusion, lifetime, target=target)
    assert charges_out.dtype == torch.float32, f'Expected charges.dtype is float32 while the output is {charges_out.dtype}.'
    assert charges_out.allclose(torch.full((locs.size(0),), 1000.), rtol=1E-6)

def test_drift_sigma_consistency():
    """Ensure sigma follows expected diffusion scaling."""
    locs = torch.tensor([1.0, 3.0])
    velocity = 2.0
    diffusion = 0.1
    lifetime = 10.0
    target = 5.0
    dt = (target - locs) / velocity

    _, _, sigma_out, _ = drift(locs, velocity, diffusion, lifetime, target=target)
    expected_sigma = torch.sqrt(2 * diffusion * dt)
    assert torch.allclose(sigma_out, expected_sigma), "Sigma should follow sqrt(2 * diffusion * dt)"

def test_drift_charge_quenching():
    """Ensure charges decrease over time for dt > 0 and are of float type."""
    locs = torch.tensor([1.0, 3.0])
    velocity = 2.0
    diffusion = 0.1
    lifetime = 10.0
    target = 5.0
    charge = torch.tensor([100, 200])

    _, _, _, charge_out = drift(locs, velocity, diffusion, lifetime, target=target, charge=charge)
    assert torch.all(charge_out <= charge), "Charge should not increase"
    assert charge_out.dtype == torch.float32, "Charge should be in float format"

def test_drift_single_dimension():
    """Check if times are updated correctly."""
    locs = torch.tensor([1.0, 3.0])
    velocity = 2.0
    diffusion = 0.1
    lifetime = 10.0
    target = 5.0
    times = torch.tensor([0.5, 1.0])

    locs, _, _, _ = drift(locs, velocity, diffusion, lifetime, target=target, times=times)
    assert len(locs.shape) == 1, f'Shape of locs should be preserved after squeezing.'
    assert locs.allclose(torch.full((len(locs),), target)), f'locs should be updated to target'

@pytest.mark.parametrize("velocity,lifetime,target,vaxis", [
    (1.0, 10.0, 5.0, 1),
    (torch.tensor(1.0), torch.tensor(10.0), torch.tensor(5.0), torch.tensor(1)),
])
def test_drift_multiple_dimensions(velocity,lifetime,target,vaxis):
    """Ensure function handles multi-dimensional locs correctly."""
    locs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    velocity = 1.0
    diffusion = torch.tensor([0.1, 0.2])
    lifetime = 10.0
    vaxis = 1
    target = 5.0

    locs_out, times_out, sigma_out, charge_out = drift(locs, velocity, diffusion, lifetime, target=target, vaxis=vaxis)
    assert locs_out.shape == locs.shape, "Shape of locs should be preserved"
    assert times_out.shape == (2,), "Times should match the number of points"
    assert sigma_out.shape == (2, 2), "Sigma should have correct shape, expected npt, vdim = {locs.shape}"
    assert charge_out.shape == (2,), "Charge should match the number of points"

if __name__ == "__main__":
    pytest.main([__file__])
