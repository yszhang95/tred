import pytest
import torch
from tred.graph import Drifter

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

def plot_drifter_simulations():
    """
    Combined plot with:
      - Upper panel: Output Time vs. Initial Location (drift simulation).
      - Lower panel: Resulting Charge vs. Initial Location (drift absorption).
    """
    # -----------------------
    # Upper Panel: Drift Times vs. Locations
    # -----------------------
    # Simulation parameters for drift times
    velocity = 2.0         # units: distance/time
    diffusion = 0.1        # units: (length^2)/time (not directly plotted)
    lifetime = 10.0        # units: time (affects absorption, not plotted)
    target = 5.0           # drift target location along the drift axis
    constant_time = 1.0    # constant initial time for all inputs
    vaxis = 0              # drift is along the first dimension
    fluctuate = False      # use deterministic mode for clarity

    # Create a Drifter instance with the given parameters.
    drifter = Drifter(diffusion=diffusion, lifetime=lifetime, velocity=velocity,
                      target=target, vaxis=vaxis, fluctuate=fluctuate)

    # Generate a range of initial locations (for example, from 0 to 10)
    initial_locs = torch.linspace(0, 10, 100)
    # Create a constant initial time tensor.
    initial_time = torch.full_like(initial_locs, constant_time)
    # Set a constant charge (e.g., 1000 electrons) for each point.
    initial_charge = torch.full_like(initial_locs, 1000, dtype=torch.float32)

    # Preserve the initial locations (input independent variable)
    input_locs = initial_locs.clone()

    # Call Drifter.forward.
    # Drifter.forward returns (dsigma, dtime, dcharge, dtail)
    dsigma, dtime, dcharge, dtail = drifter.forward(initial_time, initial_charge, initial_locs)

    # Convert tensors to numpy arrays for plotting.
    input_locs_np = input_locs.numpy()
    dtime_np = dtime.numpy()

    # -----------------------
    # Lower Panel: Charge vs. Locations
    # -----------------------
    # Simulation parameters for charge absorption
    velocity_chg = 2.0     # units: distance/time
    diffusion_chg = 0.1    # required by Drifter, though not used for theory curve
    lifetime_chg = 2.0     # absorption lifetime (time units)
    target_chg = 5.0       # drift target location along the drift axis
    Q0 = 1000.0            # initial charge (electrons)
    constant_time_chg = 0.0  # constant initial time for all inputs
    vaxis_chg = 0          # drift along the first dimension

    # Create two Drifter instances: one deterministic and one stochastic.
    drifter_det = Drifter(diffusion=diffusion_chg, lifetime=lifetime_chg, velocity=velocity_chg,
                          target=target_chg, vaxis=vaxis_chg, fluctuate=False)
    drifter_sto = Drifter(diffusion=diffusion_chg, lifetime=lifetime_chg, velocity=velocity_chg,
                          target=target_chg, vaxis=vaxis_chg, fluctuate=True)

    # Generate a range of initial locations.
    initial_locs_chg = torch.linspace(0, 10, 100)
    # Create a constant initial time for all inputs.
    initial_time_chg = torch.full_like(initial_locs_chg, constant_time_chg)
    # Create a tensor for the initial charge.
    initial_charge_chg = torch.full_like(initial_locs_chg, Q0, dtype=torch.float32)

    # Compute the theoretical drift time for each location.
    dt = (target_chg - initial_locs_chg) / velocity_chg
    # Compute the theoretical surviving charge: Q_theory = Q0 * exp(- dt / lifetime)
    Q_theory = Q0 * torch.exp(- dt / lifetime_chg)
    # For dt > 0, compute the loss fraction and standard deviation.
    L = 1 - torch.exp(- dt / lifetime_chg)
    variance = Q0 * L * (1 - L)
    std_dev = torch.sqrt(variance)
    positive_mask = dt > 0

    # Run the drift simulation using Drifter.forward.
    # forward returns (dsigma, dtime, dcharge, dtail)
    dsigma_det, dt_det, charge_det, _ = drifter_det.forward(initial_time_chg, initial_charge_chg, initial_locs_chg)
    dsigma_sto, dt_sto, charge_sto, _ = drifter_sto.forward(initial_time_chg, initial_charge_chg, initial_locs_chg)

    # Convert tensors to numpy for plotting.
    locs_np = initial_locs_chg.numpy()
    Q_theory_np = Q_theory.numpy()
    charge_det_np = charge_det.numpy()
    charge_sto_np = charge_sto.numpy()
    std_np = std_dev.numpy()

    dt_det_np = dt_det.numpy()
    dt_sto_np = dt_sto.numpy()
    dsigma_det_p1 = dsigma_det.numpy() + 1
    dsigma_sto_p1 = dsigma_sto.numpy() + 1

    # -----------------------
    # Create a figure with 3 rows, 2 columns
    #   Row 1: ax1 spans columns 1-2
    #   Row 2: ax2 spans columns 1-2
    #   Row 3: ax3 (col 1), ax4 (col 2)
    # -----------------------
    fig = plt.figure(figsize=(10, 14))

    # Top row (row=1), spanning columns 1 and 2
    ax1 = fig.add_subplot(3, 2, (1, 2))
    # Second row (row=2), spanning columns 1 and 2
    ax2 = fig.add_subplot(3, 2, (3, 4))
    # Third row, left column (row=3, col=1)
    ax3 = fig.add_subplot(3, 2, 5)
    # Third row, right column (row=3, col=2)
    ax4 = fig.add_subplot(3, 2, 6)

    # # -----------------------
    # # Create Combined Figure with Two Subplots
    # # -----------------------
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6*3))

    # Upper panel: Drift times vs. Initial Location
    ax1.plot(input_locs_np, dtime_np, 'bo-', label='Output Time')
    ax1.set_xlabel('Initial Location')
    ax1.set_ylabel('Output Time')
    ax1.set_title('Output Time vs. Initial Location (Drift Simulation)')
    # Mark the target location with a vertical dashed line.
    ax1.axvline(x=target, color='r', linestyle='--', label=f'Target = {target}')
    # Mark the constant initial time with a horizontal dashed line.
    ax1.axhline(y=constant_time, color='g', linestyle='--', label=f'Initial Time = {constant_time}')
    # Add text annotations for velocity and target information.
    ax1.text(0.05, 0.95, f'Velocity: {velocity}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top')
    ax1.text(0.05, 0.90, f'Target: {target}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top')
    ax1.legend()
    ax1.grid(True)

    # Lower panel: Charge vs. Initial Location (log scale)
    ax2.plot(locs_np, Q_theory_np, 'b-', label='Theory (Q0 exp(-dt/lifetime))')
    ax2.plot(locs_np, charge_det_np, 'ro-', label='Deterministic (fluctuate=False)', markersize=4)
    ax2.plot(locs_np, charge_sto_np, 'go-', label='Stochastic (fluctuate=True)', markersize=4)
    # Plot variance band for dt > 0.
    ax2.fill_between(locs_np[positive_mask.numpy()],
                     Q_theory_np[positive_mask.numpy()] - std_np[positive_mask.numpy()],
                     Q_theory_np[positive_mask.numpy()] + std_np[positive_mask.numpy()],
                     color='gray', alpha=0.3, label='Theory ± 1σ (for dt > 0)')
    # Mark the target location with a vertical dashed line.
    ax2.axvline(x=target_chg, color='k', linestyle='--', label=f'Target = {target_chg}')
    # Annotate with initial time, velocity, and lifetime.
    ax2.text(0.05, 0.95,
             f'Initial Time = {constant_time_chg}\nInitial charge = {Q0}'\
             f'Velocity = {velocity_chg}\nLifetime = {lifetime_chg}',
             transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax2.set_xlabel('Initial Location')
    ax2.set_ylabel('Resulting Charge [log scale]')
    ax2.set_yscale('log')
    ax2.set_title('Resulting Charge vs. Initial Location')
    ax2.legend()
    ax2.grid(True)

    mp = dt_det > 0
    ax3.plot(dt_det[mp], dsigma_det[mp].numpy(), 'ro', label='Fluctuate=False', markersize=4)
    ax3.plot(dt_sto[mp], dsigma_sto[mp].numpy(), 'g*', label='Fluctuate=True', markersize=4)
    ax3.text(0.05, 0.95,
             f'Initial Time = {constant_time_chg}\nInitial charge = {Q0}'\
             f'Velocity = {velocity_chg}\nLifetime = {lifetime_chg}',
             transform=ax3.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax3.set_title('Diffusion width vs. dt [dt>0]')
    ax3.set_xlabel('dt [log]')
    ax3.set_xscale('log')
    ax3.set_ylabel('Diffusion width [log]')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)

    mn = dt_det < 0
    ax4.plot(-dt_det[mn], dsigma_det[mn].numpy() + 1, 'ro', label='Fluctuate=False', markersize=4)
    ax4.plot(-dt_sto[mn], dsigma_sto[mn].numpy() + 1, 'g*', label='Fluctuate=True', markersize=4)
    ax4.text(0.05, 0.95,
             f'Initial Time = {constant_time_chg}\nInitial charge = {Q0}'\
             f'Velocity = {velocity_chg}\nLifetime = {lifetime_chg}',
             transform=ax4.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax4.set_title('Diffusion width vs. dt [dt<0]')
    ax4.set_xlabel('-dt [log]')
    ax4.set_xscale('log')
    ax4.set_ylabel('Diffusion width+1 [log]')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()
    fig.savefig('drifter.png')


# If running this module directly, call the plotting function.
if __name__ == '__main__':
    pytest.main([__file__])
    # plot_drift_times_vs_locs()
    # plot_charge_vs_locs()
    plot_drifter_simulations()
