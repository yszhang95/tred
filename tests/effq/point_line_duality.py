from tred.raster.steps import qline_diff3D, qpoint_diff3D
from tred.raster.steps import compute_charge_box
from tred.raster.steps import create_w_block, create_node1ds

import torch
import matplotlib.pyplot as plt


def prepare_Q(Q, X0, X1, Sigma):
    n_sigma = [3.0, 3.0, 3.0]
    grid_spacing = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64)  # 5 µm voxels
    origin = torch.tensor([0, 0, 0], dtype=torch.int64)
    offset, shape = compute_charge_box(X0, X1, Sigma, n_sigma, origin, grid_spacing)

    npoints = (2, 2, 2)
    wblock = create_w_block('gauss_legendre', npoints, grid_spacing, device=Q.device)
    x, y, z = create_node1ds('gauss_legendre', npoints, origin, grid_spacing, offset, shape)
    x = x[:, :, None, None, :, None, None]
    y = y[:, None, :, None, None, :, None]
    z = z[:, None, None, :, None, None, :]

    qline = qline_diff3D(Q, X0, X1, Sigma, x, y, z)
    qpoint = qpoint_diff3D(Q, X0, X1, Sigma, x, y, z)
    qline = qline * wblock[None, :, :, :, None, None, None]
    qpoint = qpoint * wblock[None, :, :, :, None, None, None]

    qline = qline.sum(dim=(1, 2, 3))
    qpoint = qpoint.sum(dim=(1, 2, 3))

    offset = offset * torch.as_tensor(grid_spacing)
    dimensions = shape * torch.as_tensor(grid_spacing)

    return qline, qpoint, offset, dimensions


def project_charge_to_1d(charge_3d: torch.Tensor):
    """
    Project a 3D charge field onto 1D profiles along x, y and z.

    Args:
        charge_3d: Tensor of shape (Nsteps, Nx, Ny, Nz)

    Returns:
        qx: Tensor of shape (Nsteps, Nx),  sum over y,z
        qy: Tensor of shape (Nsteps, Ny),  sum over x,z
        qz: Tensor of shape (Nsteps, Nz),  sum over x,y
    """
    if charge_3d.ndim != 4:
        raise ValueError(f"charge_3d must have shape (Nsteps, Nx, Ny, Nz), got {charge_3d.shape}")

    # Sum over the other two spatial dimensions to obtain 1D profiles
    qx = charge_3d.sum(dim=(2, 3))  # sum over y, z -> (Nsteps, Nx)
    qy = charge_3d.sum(dim=(1, 3))  # sum over x, z -> (Nsteps, Ny)
    qz = charge_3d.sum(dim=(1, 2))  # sum over x, y -> (Nsteps, Nz)

    return qx, qy, qz


def main():
    for L in [0.1, 0.5, 1.0]:
        Q = torch.tensor([1.0], dtype=torch.float64)
        X0 = torch.tensor([[-L/2, 0.0, 0.0]], dtype=torch.float64)
        X1 = torch.tensor([[L/2, 0.0, 0.0]], dtype=torch.float64)  # L segment
        Sigma = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        qline, qpoint, offset, dimensions = prepare_Q(Q, X0, X1, Sigma)
        qx_line, qy_line, qz_line = project_charge_to_1d(qline)
        qx_point, qy_point, qz_point = project_charge_to_1d(qpoint)
        diff_x = (qx_line - qx_point).abs()
        diff_y = (qy_line - qy_point).abs()
        diff_z = (qz_line - qz_point).abs()

        axis_labels = ['x', 'y', 'z']
        profiles_line = [qx_line, qy_line, qz_line]
        profiles_point = [qx_point, qy_point, qz_point]
        diffs = [diff_x, diff_y, diff_z]

        # Axes ranges for x, y, z
        starts = [offset[0, 0].item(), offset[0, 1].item(), offset[0, 2].item()]
        lengths = [dimensions[0].item(), dimensions[1].item(), dimensions[2].item()]
        counts = [diff_x.size(-1), diff_y.size(-1), diff_z.size(-1)]
        coords = [
            torch.linspace(starts[i], starts[i] + lengths[i], counts[i], dtype=torch.float64, device=diff_x.device)
            for i in range(3)
        ]

        fig, axs = plt.subplots(2, 3, figsize=(18, 8))
        for i, label in enumerate(axis_labels):
            # Top row: profile
            axs[0, i].plot(
                coords[i].cpu().numpy(),
                profiles_line[i][0].cpu().numpy(),
                label='qline'
            )
            axs[0, i].plot(
                coords[i].cpu().numpy(),
                profiles_point[i][0].cpu().numpy(),
                label='qpoint',
                linestyle='dashed'
            )
            axs[0, i].set_title(f"{label}-profile")
            axs[0, i].legend()
            axs[0, i].set_xlabel(f"{label}")
            axs[0, i].set_ylabel("Charge")

            # Bottom row: difference
            axs[1, i].plot(
                coords[i].cpu().numpy(),
                diffs[i][0].cpu().numpy(),
                label=f"|qline - qpoint|"
            )
            axs[1, i].set_title(f"{label}-diff")
            axs[1, i].legend()
            axs[1, i].set_xlabel(f"{label}")
            axs[1, i].set_ylabel("Abs Difference")

        # Create comprehensive super title
        fig.suptitle(
            f"Q={Q.tolist()}, X0={X0.tolist()}, X1={X1.tolist()}, Sigma={Sigma.tolist()}, L={L}",
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Ensure portable filename
        L_str = str(L).replace('.', 'p')
        fig.savefig(f'charge_profile_L{L_str}.png')
        plt.close(fig)


if __name__ == "__main__":
    main()
