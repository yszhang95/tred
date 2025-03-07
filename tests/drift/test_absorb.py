import torch
import pytest

import numpy as np
import matplotlib.pyplot as plt

# Make sure the 'absorb' function and Binomial class are defined/imported in the test context.
# For example, if using torch.distributions.Binomial, you may import it in the same module as absorb.

from tred.drift import absorb

def test_absorb_return_type():
    dt = torch.tensor([1.0, 2.0])
    charge = torch.tensor([100, 200])
    lifetime = 1.0
    # Expected survival: charge * exp(-dt / lifetime)
    result0 = absorb(charge, dt, lifetime, fluctuate=False)
    result1 = absorb(charge, dt, lifetime, fluctuate=True)
    assert result0.dtype == torch.float32 and result1.dtype == torch.float32, \
            f'dtype when fluctuate==False is {result0.dtype}, '\
            f'dtype when fluctuate==True is {result1.dtype}'

def test_absorb_deterministic():
    # Test the deterministic mode (fluctuate=False)
    dt = torch.tensor([1.0, 2.0])
    charge = torch.tensor([100, 200])
    lifetime = 1.0
    # Expected survival: charge * exp(-dt / lifetime)
    expected = charge * torch.exp(- dt / lifetime)
    result = absorb(charge, dt, lifetime, fluctuate=False)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

def test_absorb_dt_zero():
    # When dt is zero, no absorption should occur (both modes should return the original charge)
    dt = torch.tensor([0.0])
    charge = torch.tensor([50.])
    lifetime = 1.0
    expected = charge  # exp(0)=1 so no change
    result_det = absorb(charge, dt, lifetime, fluctuate=False)
    result_fluc = absorb(charge, dt, lifetime, fluctuate=True)
    print(expected.dtype)
    assert torch.allclose(result_det, expected), f"Deterministic: Expected {expected}, got {result_det}"
    assert torch.allclose(result_fluc, expected), f"Fluctuating: Expected {expected}, got {result_fluc}"

def test_absorb_negative_dt():
    # Negative dt should result in an increase: exp(-(-dt)/lifetime) = exp(dt/lifetime) > 1.
    dt = torch.tensor([-1.0, -2.0])
    charge = torch.tensor([100, 200])
    lifetime = 1.0
    expected = charge * torch.exp(- dt / lifetime)  # note: - dt is positive here
    result = absorb(charge, dt, lifetime, fluctuate=False)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

def test_absorb_fluctuate_range():
    # In fluctuate mode, the surviving charge should be between 0 and the original charge.
    dt = torch.tensor([1.0, 2.0])
    charge = torch.tensor([100, 200])
    lifetime = 1.0
    result = absorb(charge, dt, lifetime, fluctuate=True)
    # Verify each element is at least 0 and does not exceed the initial charge.
    assert torch.all(result >= 0), f"Some values are below 0: {result}"
    assert torch.all(result <= charge), f"Some values exceed the initial charge: {result}"

def test_absorb_fluctuate_mean():
    # Test that the average over many samples is close to the deterministic expectation.
    dt = torch.tensor([1.0, 2.0])
    charge = torch.tensor([100, 200])
    lifetime = 1.0
    expected = charge * torch.exp(- dt / lifetime)

    num_samples = 10000
    samples = []
    for _ in range(num_samples):
        samples.append(absorb(charge, dt, lifetime, fluctuate=True))
    samples = torch.stack(samples)  # shape: (num_samples, npts)
    sample_mean = torch.mean(samples.float(), dim=0)

    # Allow some tolerance given the stochastic fluctuations.
    assert torch.allclose(sample_mean, expected.float(), rtol=0.1, atol=1.0), (
        f"Expected mean approx. {expected}, got {sample_mean}"
    )


# ================== Plotting Function ==================
def plot_charge_distributions():
    # Create a figure with two subplots (axes)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # ------------------ First Plot: Charge Distributions ------------------
    num_samples = 10000
    initial_charge = torch.tensor([100])
    dt_values = [-0.5, 0, 0.5, 1.0, 2.0, 3.0]
    lifetime = 1.0

    cmap = plt.get_cmap('viridis', len(dt_values))

    for i, dt in enumerate(dt_values):
        dt_tensor = torch.tensor([dt])
        # Vectorized simulation: repeat initial charge and dt for num_samples trials
        result = absorb(initial_charge.repeat(num_samples), dt_tensor.repeat(num_samples),
                        lifetime, fluctuate=True)
        samples = result.tolist()

        color = cmap(i)
        ax1.hist(samples, bins=110, range=(-0.5, 109.5), alpha=0.5,
                 label=f'dt = {dt}', color=color)

        # Deterministic expected surviving charge (fluctuate=False)
        expected = absorb(initial_charge, dt_tensor, lifetime, fluctuate=False).item()
        # Mean of the stochastic samples
        mean_sample = np.mean(samples)

        # Compute loss fraction and theoretical variance for dt > 0, else loss=1.
        if dt > 0:
            loss = 1 - (expected / initial_charge.item())
        else:
            loss = 1.0
        theoretical_variance = initial_charge.item() * loss * (1 - loss)
        measured_variance = np.var(samples, ddof=1)

        # Add text annotation with variances
        ax1.text(0.05, 0.95 - i*0.05,
                 f"dt:{dt:.1f}, var (theory): {theoretical_variance:02.2f}, var (measured): {measured_variance:02.2f}",
                 transform=ax1.transAxes, fontsize=12, color='black')

        # Vertical lines for deterministic expectation and sample mean
        ax1.axvline(x=expected, color=color, linestyle='--', linewidth=2,
                    label=f'Expected (dt={dt})')
        ax1.axvline(x=mean_sample, color=color, linestyle='-', linewidth=2,
                    label=f'Mean (dt={dt})')
    ax1.text(0.15, 0.5, f'initial charge {initial_charge.item()}; lifetime {lifetime}',
             transform=ax1.transAxes, fontsize=12, color='black')

    ax1.set_xlabel('Surviving Charge')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Charge Distributions for Different dt Values')
    ax1.legend()

    # ------------------ Second Plot: Surviving Charge vs. dt ------------------
    # Parameters for second plot (using a different initial charge)
    lifetime = 1.0
    initial_charge2 = torch.full((100,), 100)
    dt_values_line = torch.linspace(0, 5, len(initial_charge2))
    dt_curve = torch.linspace(0, 5, 1000)

    deterministic_points = absorb(initial_charge2, dt_values_line, lifetime, fluctuate=False)
    stochastic_points = absorb(initial_charge2, dt_values_line, lifetime, fluctuate=True)
    theory_curve = torch.full((len(dt_curve),), initial_charge2[0].item()) * torch.exp(-dt_curve/lifetime)

    neg_dt_values = torch.linspace(-0.5, 0, 5)
    neg_dt_curve = torch.linspace(-0.5, 0, 200)
    neg_det_points = absorb(torch.full((len(neg_dt_values),), initial_charge2[0].item()),
                             neg_dt_values, lifetime, fluctuate=False)
    neg_sto_points = absorb(torch.full((len(neg_dt_values),), initial_charge2[0].item()),
                             neg_dt_values, lifetime, fluctuate=True)
    neg_the_curve = torch.full((len(neg_dt_curve),), initial_charge2[0].item()) * torch.exp(-neg_dt_curve/lifetime)

    # Compute error (vertical range) for the theory curve
    error_curve = torch.full((len(dt_curve),), initial_charge2[0].item()) * \
                  torch.exp(-dt_curve/lifetime) * (1 - torch.exp(-dt_curve/lifetime))
    error_curve = np.sqrt(error_curve)

    # Plot negative dt part of theory curve
    ax2.plot(neg_dt_curve.numpy(), neg_the_curve.numpy(), 'k-', linewidth=2)
    # Plot positive dt theory curve and shaded uncertainty
    ax2.plot(dt_curve.numpy(), theory_curve.numpy(), 'k-', linewidth=2, label='Theory: N₀ exp(-dt/lifetime)')
    ax2.fill_between(dt_curve.numpy(),
                     (theory_curve - error_curve).numpy(),
                     (theory_curve + error_curve).numpy(),
                     color='gray', alpha=0.3, label='Theory ± expected binomial variance')
    # Plot deterministic and stochastic points
    ax2.plot(dt_values_line.numpy(), deterministic_points.numpy(), 'bo', label='Fluctuate=False')
    ax2.plot(neg_dt_values.numpy(), neg_det_points.numpy(), 'bo')
    ax2.plot(dt_values_line.numpy(), stochastic_points.numpy(), 'ro', label='Fluctuate=True')
    ax2.plot(neg_dt_values.numpy(), neg_sto_points.numpy(), 'ro')

    ax2.text(0.65, 0.5, f'initial charge {initial_charge2[0].item()}; lifetime {lifetime}',
             transform=ax2.transAxes, fontsize=12, color='black')

    ax2.set_xlabel('dt')
    ax2.set_ylabel('Surviving Charge')
    ax2.set_title('Surviving Charge vs. dt')
    ax2.legend()

    fig.tight_layout()
    plt.show()
    fig.savefig('absorb.png')


# ================== Example Execution ==================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])

    # Plot distributions for given dt values.
    plot_charge_distributions()
