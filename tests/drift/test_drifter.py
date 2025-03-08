import pytest
import torch
from tred.drift import transport
from tred.graph import Drifter

import numpy as np
import matplotlib.pyplot as plt

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


@pytest.mark.parametrize("locs, times, velocity, target, tshift, expected", [
    # Provided times, scalar tshift.
    (torch.tensor([1.0, 3.0]), torch.tensor([0.5, 1.0]), 2.0, 5.0, 0.2,
     torch.tensor([0.5, 1.0]) + transport(torch.tensor([1.0, 3.0]), 5.0, 2.0) - 0.2),
    # No initial times, scalar tshift.
    (torch.tensor([1.0, 3.0]), None, 2.0, 5.0, 0.2,
     transport(torch.tensor([1.0, 3.0]), 5.0, 2.0) - 0.2),
    # No initial times, 0D tensor tshift.
    (torch.tensor([1.0, 3.0]), None, 2.0, 5.0, torch.tensor(0.2),
     transport(torch.tensor([1.0, 3.0]), 5.0, 2.0) - 0.2),
])
def test_drifter_tshift_correction(locs, times, velocity, target, tshift, expected):
    """
    Test that Drifter applies the tshift correction correctly.
    The output times should be: initial times + dt - tshift.
    """
    diffusion = 0.1
    lifetime = 10.0
    charge = None  # use default charge
    drifter = Drifter(diffusion, lifetime, velocity, target, vaxis=0, tshift=tshift)
    # Pass tshift as a keyword argument (requires Drifter.forward to forward it)
    dsigma, dtime, dcharge, dlocs = drifter.forward(times, charge, locs)
    assert torch.allclose(dtime, expected), f"Expected times {expected}, got {dtime}"


@pytest.mark.parametrize("locs, times, velocity, target, drtoa, expected", [
    # Provided times, scalar drtoa.
    (torch.tensor([1.0, 3.0]), torch.tensor([0.5, 1.0]), 2.0, 5.0, 4.0,
     torch.tensor([0.5, 1.0]) + transport(torch.tensor([1.0, 3.0]), 5.0, 2.0) - (4.0 / 2.0)),
    # Provided times, 0D tensor drtoa.
    (torch.tensor([1.0, 3.0]), torch.tensor([0.5, 1.0]), 2.0, 5.0, torch.tensor(4.0),
     torch.tensor([0.5, 1.0]) + transport(torch.tensor([1.0, 3.0]), 5.0, 2.0) - (torch.tensor(4.0) / 2.0)),
    # No initial times, scalar drtoa.
    (torch.tensor([1.0, 3.0]), None, 2.0, 5.0, 4.0,
     transport(torch.tensor([1.0, 3.0]), 5.0, 2.0) - (4.0 / 2.0)),
    # No initial times, 0D tensor drtoa.
    (torch.tensor([1.0, 3.0]), None, 2.0, 5.0, torch.tensor(4.0),
     transport(torch.tensor([1.0, 3.0]), 5.0, 2.0) - (torch.tensor(4.0) / 2.0)),
])
def test_drifter_drtoa_correction(locs, times, velocity, target, drtoa, expected):
    """
    Test that Drifter applies the drtoa correction correctly.
    The output times should be: initial times + dt - (drtoa/velocity).
    """
    diffusion = 0.1
    lifetime = 10.0
    charge = None  # use default charge
    drifter = Drifter(diffusion, lifetime, velocity, target, vaxis=0, drtoa=drtoa)
    dsigma, dtime, dcharge, dlocs = drifter.forward(times, charge, locs)
    assert torch.allclose(dtime, expected), f"Expected times {expected}, got {dtime}"


@pytest.mark.parametrize("locs, times, velocity, target, tshift", [
    # Provided times, scalar tshift.
    (torch.tensor([1.0, 3.0]), torch.tensor([0.5, 1.0]), 2.0, 5.0, 0.2),
    # No initial times.
    (torch.tensor([1.0, 3.0]), None, 2.0, 5.0, 0.2),
])
def test_drifter_tshift_interference(locs, times, velocity, target, tshift):
    """
    Check that providing tshift only affects the time output and does not alter sigma,
    charge, or the updated locs.
    """
    diffusion = 0.1
    lifetime = 10.0
    charge = None
    drifter_with = Drifter(diffusion, lifetime, velocity, target, vaxis=0, tshift=tshift)
    drifter_without = Drifter(diffusion, lifetime, velocity, target, vaxis=0)
    out_with = drifter_with.forward(times, charge, locs)
    out_without = drifter_without.forward(times, charge, locs)
    # Compare sigma, charge, and tail locs remain unchanged.
    assert torch.allclose(out_with[0], out_without[0]), "Sigma should be unaffected by tshift."
    assert torch.allclose(out_with[3], out_without[3]), "Tail locs should be unaffected by tshift."
    assert torch.allclose(out_with[2], out_without[2]), "Charge should be unaffected by tshift."


@pytest.mark.parametrize("locs, times, velocity, target, drtoa", [
    # Provided times, scalar drtoa.
    (torch.tensor([1.0, 3.0]), torch.tensor([0.5, 1.0]), 2.0, 5.0, 4.0),
    # No initial times.
    (torch.tensor([1.0, 3.0]), None, 2.0, 5.0, 4.0),
])
def test_drifter_drtoa_interference(locs, times, velocity, target, drtoa):
    """
    Check that providing drtoa only affects the time output and does not alter sigma,
    charge, or the updated locs.
    """
    diffusion = 0.1
    lifetime = 10.0
    charge = None
    drifter_with = Drifter(diffusion, lifetime, velocity, target, vaxis=0, drtoa=drtoa)
    drifter_without = Drifter(diffusion, lifetime, velocity, target, vaxis=0, drtoa=drtoa)
    out_with = drifter_with.forward(times, charge, locs)
    out_without = drifter_without.forward(times, charge, locs)
    # Compare sigma, charge, and tail locs remain unchanged.
    assert torch.allclose(out_with[0], out_without[0]), "Sigma should be unaffected by drtoa."
    assert torch.allclose(out_with[3], out_without[3]), "Tail locs should be unaffected by drtoa."
    assert torch.allclose(out_with[2], out_without[2]), "Charge should be unaffected by drtoa."


def test_drifter_mutually_exclusive_tshift_and_drtoa():
    """
    Test that Drifter.forward raises an error when both tshift and drtoa are provided.
    """
    locs = torch.tensor([1.0, 3.0])
    diffusion = 0.1
    velocity = 2.0
    lifetime = 10.0
    target = 5.0
    times = torch.tensor([0.5, 1.0])
    tshift = 0.2
    drtoa = 4.0
    charge = None
    with pytest.raises(ValueError):
        drifter = Drifter(diffusion, lifetime, velocity, target, vaxis=0, tshift=tshift, drtoa=drtoa)


def plot_drift_time_vs_location():
    """
    Figure 1: Plots Output Time vs. Initial Location for drift simulation.
    Now includes tshift and drtoa corrections on the same axis.
    """
    # Simulation parameters
    velocity = 2.0
    diffusion = 0.1
    lifetime = 10.0
    target = 5.0
    constant_time = 1.0
    tshift_value = 0.5  # Example correction shift in time
    drtoa_value = 2.0   # Example drift displacement correction

    # Initialize Drifter
    drifter = Drifter(diffusion=diffusion, lifetime=lifetime, velocity=velocity,
                      target=target, vaxis=0, fluctuate=False)
    drifter_tshift = Drifter(diffusion=diffusion, lifetime=lifetime, velocity=velocity,
                      target=target, vaxis=0, fluctuate=False, tshift=tshift_value, drtoa=None)
    drifter_drtoa = Drifter(diffusion=diffusion, lifetime=lifetime, velocity=velocity,
                      target=target, vaxis=0, fluctuate=False, tshift=None, drtoa=drtoa_value)

    # Generate initial locations
    initial_locs = torch.linspace(0, 10, 100)
    initial_time = torch.full_like(initial_locs, constant_time)
    initial_charge = torch.full_like(initial_locs, 1000, dtype=torch.float32)

    # Compute drift time for different cases
    _, dtime_no_correction, _, _ = drifter.forward(initial_time, initial_charge, initial_locs)
    _, dtime_tshift, _, _ = drifter_tshift.forward(initial_time, initial_charge, initial_locs)
    _, dtime_drtoa, _, _ = drifter_drtoa.forward(initial_time, initial_charge, initial_locs)

    # Convert tensors to numpy for plotting
    input_locs_np = initial_locs.numpy()
    dtime_no_correction_np = dtime_no_correction.numpy()
    dtime_tshift_np = dtime_tshift.numpy()
    dtime_drtoa_np = dtime_drtoa.numpy()

    # -----------------------
    # Create a single-axis figure
    # -----------------------
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot all three drift time variations on one axis
    ax1.plot(input_locs_np, dtime_no_correction_np, 'bo-', label='No Correction')
    ax1.plot(input_locs_np, dtime_tshift_np, 'r--', label=f'tshift = {tshift_value}')
    ax1.plot(input_locs_np, dtime_drtoa_np, 'g-.', label=f'drtoa = {drtoa_value}')

    # Reference lines
    ax1.axvline(x=target, color='black', linestyle='--', label=f'Target = {target}')
    ax1.axhline(y=constant_time, color='gray', linestyle='--', label=f'Initial Time = {constant_time}')

    # Labels and title
    ax1.set_xlabel('Initial Location')
    ax1.set_ylabel('Output Time')
    ax1.set_title('Output Time vs. Initial Location (Drift Simulation)')
    ax1.legend()
    ax1.grid(True)

    # -----------------------
    # Add velocity text annotation
    # -----------------------
    text_x = 0.05  # Position relative to axis
    text_y = 0.90  # Position relative to axis
    ax1.text(text_x, text_y, f'Velocity = {velocity} units/time',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Show plot
    plt.show()
    plt.savefig('drifter_dt.png')


def plot_charge_vs_location():
    """
    Figure 2: Plots Resulting Charge vs. Initial Location for drift absorption.
    """
    # Simulation parameters
    velocity = 2.0
    diffusion = 0.1
    lifetime = 2.0
    target = 5.0
    Q0 = 1000.0

    drifter_det = Drifter(diffusion=diffusion, lifetime=lifetime, velocity=velocity, target=target, vaxis=0, fluctuate=False)
    drifter_sto = Drifter(diffusion=diffusion, lifetime=lifetime, velocity=velocity, target=target, vaxis=0, fluctuate=True)

    initial_locs = torch.linspace(0, 10, 100)
    initial_time = torch.full_like(initial_locs, 0.0)
    initial_charge = torch.full_like(initial_locs, Q0, dtype=torch.float32)

    dt = (target - initial_locs) / velocity
    Q_theory = Q0 * torch.exp(- dt / lifetime)
    positive_mask = dt > 0
    variance = Q0 * (1 - torch.exp(- dt / lifetime)) * torch.exp(- dt / lifetime)
    std_dev = torch.sqrt(variance)

    _, _, charge_det, _ = drifter_det.forward(initial_time, initial_charge, initial_locs)
    _, _, charge_sto, _ = drifter_sto.forward(initial_time, initial_charge, initial_locs)

    slope_det, _ = np.polyfit(dt[positive_mask].numpy(),
                                  np.log(charge_det[positive_mask].numpy()), 1)
    slope_sto, _ = np.polyfit(dt[positive_mask].numpy(),
                                  np.log(charge_sto[positive_mask].numpy()), 1)

    plt.figure(figsize=(8, 6))
    plt.plot(initial_locs.numpy(), Q_theory.numpy(), 'b-', label='Theory (Q0 exp(-dt/lifetime))')
    plt.plot(initial_locs.numpy(), charge_det.numpy(), 'ro-', label='Deterministic', markersize=4)
    plt.plot(initial_locs.numpy(), charge_sto.numpy(), 'g*-', label='Stochastic', markersize=4)
    plt.fill_between(initial_locs.numpy()[positive_mask.numpy()],
                     (Q_theory - std_dev)[positive_mask].numpy(),
                     (Q_theory + std_dev)[positive_mask].numpy(),
                     color='gray', alpha=0.3, label='Theory ± 1σ')


    plt.axvline(x=target, color='k', linestyle='--', label=f'Target = {target}')
    plt.xlabel('Initial Location')
    plt.ylabel('Resulting Charge [log scale]')
    plt.yscale('log')
    plt.title('Resulting Charge vs. Initial Location')
    plt.legend()

    ax1 = plt.gca()
    text_x = 0.05
    text_y = 0.95
    ax1.text(text_x, text_y, f'Exponent from fit, deterministic, {slope_det:.3f}\n'
             f'Exponent from fit, stochastic, {slope_sto:.3f}\n'
             f'1/lifetime (fixed configuration)  {1./lifetime:.3f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.grid(True)
    plt.show()

    plt.savefig('drifter_charge.png')


def plot_diffusion_width():
    """
    Figure 3: Plots Diffusion Width vs. Drift Time with two subplots (dt > 0 and dt < 0).
    """
    # Simulation parameters
    velocity = 1.0
    diffusion = 0.1
    lifetime = 2.0
    target = 5.0
    Q0 = 1000.0

    drifter_det = Drifter(diffusion=diffusion, lifetime=lifetime, velocity=velocity, target=target, vaxis=0, fluctuate=False)
    drifter_sto = Drifter(diffusion=diffusion, lifetime=lifetime, velocity=velocity, target=target, vaxis=0, fluctuate=True)

    initial_locs = torch.linspace(0, 10, 100)
    initial_time = torch.full_like(initial_locs, 0.0)
    initial_charge = torch.full_like(initial_locs, Q0, dtype=torch.float32)

    dsigma_det, dt_det, _, _ = drifter_det.forward(initial_time, initial_charge, initial_locs)
    dsigma_sto, dt_sto, _, _ = drifter_sto.forward(initial_time, initial_charge, initial_locs)

    positive_mask = dt_det > 0
    negative_mask = dt_det < 0

    slope, intercept = np.polyfit(np.log(dt_det[positive_mask].numpy()),
                                  np.log(dsigma_det[positive_mask].numpy()), 1)

    # Create a figure with two subplots
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

    # Left panel: dt > 0
    ax3.plot(dt_det[positive_mask].numpy(), dt_det[positive_mask].numpy()**slope*np.exp(intercept), '--', label=f'Fit, Power {slope:.2f}')
    ax3.plot(dt_det[positive_mask].numpy(), dsigma_det[positive_mask].numpy(), 'ro', label='Fluctuate=False', markersize=4)
    ax3.plot(dt_sto[positive_mask].numpy(), dsigma_sto[positive_mask].numpy(), 'g*', label='Fluctuate=True', markersize=4)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('dt [log]')
    ax3.set_ylabel('Diffusion width [log]')
    ax3.set_title('Diffusion Width vs. dt [dt > 0]')
    ax3.legend()
    ax3.grid(True)

    # Right panel: dt < 0
    ax4.plot(-dt_det[negative_mask].numpy(), dsigma_det[negative_mask].numpy() + 1, 'ro', label='Fluctuate=False', markersize=4)
    ax4.plot(-dt_sto[negative_mask].numpy(), dsigma_sto[negative_mask].numpy() + 1, 'g*', label='Fluctuate=True', markersize=4)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('-dt [log]')
    ax4.set_ylabel('Diffusion width + 1 [log]')
    ax4.set_title('Diffusion Width vs. dt [dt < 0]')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig('drifter_diffwidth.png')


# If running this module directly, call the plotting function.
if __name__ == '__main__':
    pytest.main([__file__])
    plot_drift_time_vs_location()
    plot_charge_vs_location()
    plot_diffusion_width()
