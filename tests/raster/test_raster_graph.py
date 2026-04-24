from tred.graph import Drifter, Raster
import matplotlib.pyplot as plt
import torch
import numpy as np

def flat_single_rasterized_data_2d(location, data):
    '''
    location: 3-vector
    data: np.ndarray, 2D array
    '''
    print(location, data.shape)
    loc_out = (location[0] + np.arange(data.shape[0]))[:,None] \
        + (location[1]+ np.arange(data.shape[1]))[None,:]
    return loc_out, data

def test_drifter_raster():
    print("Running test_drifter_raster...")

    # Define common parameters
    diffusion_val = [0.2, 0.2, 0.2]  # units [length^2]/[time]
    lifetime_val = 100.0  # units [time]
    target_anode = 10.0   # units [length], anode at z=0

    # Test cases for drift direction
    velocities = [2.0, -2.0] # positive and negative drift velocity
    velocity_labels = ["Positive Drift (v > 0)", "Negative Drift (v < 0)"]

    # Data for 2D: 1D pixel (y) + 1D time (t)
    # Let y be axis 0, and z (drift dimension) be axis 1.
    # The raster will transform z (distance) to t (time).

    # Initial time of event
    initial_time = torch.tensor([0.0, 0.0], dtype=torch.float32) # units [time]
    # Initial charge of event
    charge = torch.tensor([1.0, 1.0], dtype=torch.float32) # arbitrary units

    # Case 2: Steps (line-like charges)
    initial_tail_steps = torch.tensor([[1.0, 5.0, 2], [2.0, 8.0, 12]], dtype=torch.float32)
    initial_head_steps = torch.tensor([[2.0, 3.0, 5], [3.0, 6.0, 15]], dtype=torch.float32)

    # Raster parameters
    grid_spacing = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32) # [y_spacing, t_spacing]
    pdims = (0,1) # x,y-dimension as pixel dimension
    tdim = 2     # t-dimension (originally z-dimension) as time dimension in raster output

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,16), sharey=True)

    for i, velocity_val in enumerate(velocities):
        label = velocity_labels[i]
        print(f"\n--- Testing with {label} ---")

        # Initialize Drifter
        drifter = Drifter(diffusion=diffusion_val, lifetime=lifetime_val, velocity=velocity_val,
                          target=target_anode, vaxis=2) # vaxis=2 means drift along z

        # Test Steps
        print(f"  Drifting Steps...")
        dsigma_steps, dtime_steps, dcharge_steps, dtail_steps, dhead_steps = drifter.forward(
            time=torch.tensor([initial_time[i]]),
            charge=torch.tensor([charge[i]]),
            tail=initial_tail_steps[i].unsqueeze(0),
            head=initial_head_steps[i].unsqueeze(0)
        )
        print(f"    dtail_steps (y, z_drifted_tail): {dtail_steps.detach().cpu().numpy()}")
        print(f"    dhead_steps (y, z_drifted_head): {dhead_steps.detach().cpu().numpy()}")
        print(f"    dtime_steps (drifted time for tail): {dtime_steps.detach().cpu().numpy()}")
        print(f"    dsigma_steps (diffusion width): {dsigma_steps.detach().cpu().numpy()}")
        print(f"    dcharge_steps (attenuated charge): {dcharge_steps.detach().cpu().numpy()}")

        # Initialize Raster for Steps
        raster_steps_obj = Raster(velocity=velocity_val, grid_spacing=grid_spacing, pdims=pdims, tdim=tdim)
        rastered_block_steps = raster_steps_obj.forward(
            sigma=dsigma_steps,
            time=dtime_steps,
            charge=dcharge_steps,
            tail=dtail_steps,
            head=dhead_steps
        )


        # Define the specific forward and inverse functions for this plot
        # We use lambda functions to "bake in" target_z and velocity
        raster_data_steps = rastered_block_steps.data.cpu().numpy()
        raster_data_steps = np.sum(raster_data_steps, axis=(0, 1))
        raster_location_steps = np.squeeze(rastered_block_steps.location.cpu().numpy())[1:]

        y_min_raster_steps = raster_location_steps[0] * grid_spacing[0].item()
        y_max_raster_steps = (raster_location_steps[0] + raster_data_steps.shape[0]) * grid_spacing[0].item()
        t_min_raster_steps = raster_location_steps[1] * grid_spacing[1].item()
        t_max_raster_steps = (raster_location_steps[1] + raster_data_steps.shape[1]) * grid_spacing[1].item()
        raster_location_steps, raster_data_steps = flat_single_rasterized_data_2d(raster_location_steps, raster_data_steps)

        axes[i,0].plot([initial_tail_steps[i, 2].item(), initial_head_steps[i, 2].item()],
                   [initial_tail_steps[i, 1].item(), initial_head_steps[i, 1].item()], label='step')
        axes[i,0].axvline(x=target_anode, color='gray', linestyle='--', label=f'Anode (z={target_anode})')
        axes[i,0].set_title(velocity_labels[i])
        axes[i,1].imshow(raster_data_steps.T, origin='lower', aspect='auto',
                   extent=[t_min_raster_steps, t_max_raster_steps, y_min_raster_steps, y_max_raster_steps],
                   cmap='plasma', alpha=0.8)

        axes[i,0].legend()


    fig.canvas.draw()
    plt.close(fig)
    print("Test finished.")
