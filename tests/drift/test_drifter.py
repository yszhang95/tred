import pytest
import torch
from tred.graph import Drifter

@pytest.mark.parametrize("diffusion,lifetime,velocity,target,expect_error", [
    (None, 10.0, 1.0, 0.0, True),      # diffusion is None
    (0.1, None, 1.0, 0.0, True),       # lifetime is None
    (0.1, 10.0, None, 0.0, True),      # velocity is None
    (0.1, 10.0, 1.0, None, True),      # target is None
    (0.1, 10.0, 1.0, 5.0, False),      # all valid
    (torch.tensor([0.1, 0.1, 0.1]), 10.0, 1.0, 5.0, False),      # all valid
])
def test_drifter_init_validation(diffusion, lifetime, velocity, target, expect_error):
    """
    Test that Drifter raises ValueError if any required argument is missing,
    and constructs successfully otherwise.
    """
    if expect_error:
        with pytest.raises(ValueError):
            Drifter(diffusion, lifetime, velocity, target=target)
    else:
        d = Drifter(diffusion, lifetime, velocity, target=target)
        assert isinstance(d, Drifter)


def test_drifter_forward_no_head():
    """
    Test Drifter forward pass (no head). Check shapes and basic value correctness.
    """
    # Construct Drifter
    diffusion = 0.1
    lifetime = 10.0
    velocity = 2.0
    target = 5.0
    vaxis = 0
    drifter = Drifter(diffusion, lifetime, velocity, target=target, vaxis=vaxis)

    # Prepare inputs
    locs = torch.tensor([1.0, 3.0])
    time = torch.tensor([0.0, 1.0])  # initial times
    charge = torch.tensor([100.0, 200.0])

    # Run forward
    dsigma, dtime, dcharge, dtail = drifter(time, charge, locs)

    # Check shapes
    assert dsigma.shape == locs.shape, "Sigma should match shape of locs (after drifting)."
    assert dtime.shape == time.shape, "Time output should match the input shape."
    assert dcharge.shape == charge.shape, "Charge output should match input shape."
    assert dtail.shape == locs.shape, "Tail output should match input locs shape."

    # Check that tail is set to target along vaxis
    assert torch.allclose(dtail, torch.full_like(locs, target)), (
        f"Drifter should set locs along axis={vaxis} to {target}"
    )

    # Check that time is incremented by dt
    dt = (target - locs) / velocity
    assert torch.allclose(dtime, time + dt), "Time should be incremented by drift time."

    # Check that charge is reduced (exponential factor)
    expected_charge = charge * torch.exp(-dt / lifetime)
    assert torch.allclose(dcharge, expected_charge), "Charge should follow exponential quenching."


def test_drifter_forward_with_head():
    """
    Test Drifter forward pass when a head location is provided.
    Drifter should then return 5 outputs: (dsigma, dtime, dcharge, dtail, dhead).
    """
    # Construct Drifter
    #diffusion = torch.tensor([0.2, 0.2])
    diffusion = [0.2, 0.2]
    lifetime = 5.0
    velocity = 1.0
    target = 0.0
    drifter = Drifter(diffusion, lifetime, velocity, target=target, vaxis=1)

    # Prepare inputs (2D)
    tail = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    head = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    time = torch.tensor([0.1, 1.2])
    charge = torch.tensor([300.0, 400.0])

    # Run forward
    outputs = drifter(time, charge, tail, head=head)
    assert len(outputs) == 5, "When 'head' is given, Drifter should return five outputs."
    dsigma, dtime, dcharge, dtail, dhead = outputs

    # Check shapes
    assert dsigma.shape == tail.shape, "Sigma shape should match shape of tail."
    assert dtime.shape == time.shape, "Time shape should match the shape of input times."
    assert dcharge.shape == charge.shape, "Charge shape should match the shape of input charge."
    assert dtail.shape == tail.shape, "dtail shape should match tail."
    assert dhead.shape == head.shape, "dhead shape should match head."

    # For vaxis=1, we expect tail[:,1] to become target
    assert torch.allclose(dtail[:, 1], torch.full((2,), target)), (
        "Drifter should set locs along axis=1 to target"
    )

    # dhead = head + (dtail - tail) => it shifts "head" by the same amount that "tail" was shifted
    expected_dhead = head + (dtail - tail)
    assert torch.allclose(dhead, expected_dhead), (
        "dhead should reflect the same shift from tail -> dtail"
    )


def test_drifter_negative_velocity():
    """
    Test negative velocity scenario. If velocity is negative but target - loc > 0,
    the drift time is negative, so sigma should shrink or go to zero.
    """
    drifter = Drifter(diffusion=0.1, lifetime=10.0, velocity=-2.0, target=5.0)
    locs = torch.tensor([2.0, 3.0])
    time = torch.tensor([0.0, 0.0])
    charge = torch.tensor([1000.0, 1000.0])

    dsigma, dtime, dcharge, dtail = drifter(time, charge, locs)

    # dt should be negative => times get incremented by negative dt
    assert torch.all(dtime < 0), (
        f"For negative velocity with this setup, drift times are negative: got {dtime}."
    )
    # Sigma for negative dt is forced to zero in underlying drift code
    assert torch.allclose(dsigma, torch.tensor(0.0)), "Sigma should vanish (or clamp to zero) for negative dt."
    # Charge for negative dt can actually go up if using pure drift math, but
    # here we check that it is not forcibly increased. The drift() function
    # uses an exponential that can increase for negative dt. If it’s clamped,
    # we’d see an increase, but test that it's not decreased in an unexpected way:
    assert torch.all(dcharge >= 1000.0), (
        f"For negative dt, charge should not be forcibly decreased. Got {dcharge}."
    )
    # dtail should be set to the target, though it's physically contradictory
    # that we ended up with negative dt. The drift() function straightforwardly
    # sets locs[vaxis] to target.
    assert torch.allclose(dtail, torch.full_like(locs, 5.0)), (
        "Tail location should still be forced to target in code."
    )


def test_drifter_fluctuate():
    """
    Test that Drifter's fluctuate argument is properly registered.
    We won't do a deep check of statistical distribution, but we can
    check that the Drifter passes along the flag without error.
    """
    drifter = Drifter(diffusion=0.1, lifetime=10.0, velocity=2.0,
                      target=5.0, vaxis=0, fluctuate=True)
    locs = torch.tensor([1.0, 2.0, 3.0])
    time = torch.tensor([0.0, 0.5, 1.0])
    charge = torch.tensor([100.0, 200.0, 300.0])

    # Because fluctuate=True triggers a Binomial draw inside drift(),
    # the result is random. We'll just make sure it runs and returns
    # the correct shape.
    dsigma, dtime, dcharge, dtail = drifter(time, charge, locs)

    assert dsigma.shape == locs.shape
    assert dtime.shape == time.shape
    assert dcharge.shape == charge.shape
    assert dtail.shape == locs.shape
    # We expect at most the same or smaller charge values:
    #  There's a random variation, but on average it should be less.
    assert torch.all(dcharge <= charge), "Fluctuating charges should not exceed initial values."


def test_drifter_vaxis_out_of_range():
    """
    If vaxis is out of range for the given locs, the underlying drift
    function should raise an error.
    """
    with pytest.raises(ValueError, match="illegal vector axis"):
        drifter = Drifter(diffusion=0.1, lifetime=10.0, velocity=1.0, target=0.0, vaxis=10)
        drifter(None, None, torch.tensor([1., 2.]))
