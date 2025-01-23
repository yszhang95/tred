import torch

import numpy as np
import scipy as sp

import json

def make_tensor(source, device, dtype=torch.float32):
    '''Aliasing or create a tensor if not existing.
    Result tensor will be moved to the device
    '''
    if isinstance(source, torch.Tensor):
        t = source
        t = t.to(device)
        t.requires_grad = False
        if dtype != t.dtype:
            raise ValueError(f'Wrong dtype given. The dtype of source is {t.dtype}')
    else:
        t = torch.tensor(source, dtype=dtype, requires_grad=False,
                         device=device)
    return t

class UniversalGrid:
    """
    A class representing a universal grid in N dimensions. The grid is defined by its
    origin and grid spacing along each dimension.

    This class supports operations such as converting between physical coordinates and grid
    indices, and it accommodates an optional origin offset for shifting the reference point.

    TBD: alignment for origin with some reference, like edges of pixels

    Attributes:
        _grid_spacing (torch.Tensor): Grid spacing along each dimension.
        _origin (torch.Tensor): Physical coordinates of the grid's origin.
    """

    def __init__(self, grid_spacing=None, origin=None, device='cpu'):
        """
        Initializes the grid using separate arrays for each dimension.

        Args:
            grid_spacing (tuple): Spacing between grid points in each dimension.
            origin (tuple, optional): Physical coordinates of the grid's origin. If None,
                the origin defaults to `min_limits`. If provided, the origin must lie within
                the grid bounds.

        Raises:
            NotImplementedError: If the provided origin is outside the minimum limits.
        """
        self._grid_spacing = torch.tensor(grid_spacing, device=device,
                                          dtype=torch.float32, requires_grad=False) if grid_spacing else None
        self._origin = torch.tensor(origin, device=device,
                                    dtype=torch.float32, requires_grad=False) if origin else None
        self._device = device
    @property
    def origin(self):
        return self._origin

    @property
    def grid_spacing(self):
        return self._grid_spacing

    @property
    def device(self):
        return self._device

    @staticmethod
    def compute_coordinate(idxs, spacing, origin, device='cpu'):
        '''
        idxs : (N, vdim); index of grid point
        spacing : (vdim, )
        origin : (vdim,)
        return origin + spacing * idx
        '''
        idxs = make_tensor(idxs, device=device, dtype=torch.int64)
        if len(idxs.shape) == 1:
            idxs = idxs.unsqueeze(0)
        origin = make_tensor(origin, device=device, dtype=torch.float32)
        spacing = make_tensor(spacing, device=device, dtype=torch.float32)
        return origin.unsqueeze(0) + idxs * spacing.unsqueeze(0)

    @staticmethod
    def compute_index(coords, origin, spacing, device='cpu'):
        '''
        spacing : (vdim, )
        coords : (N, vdim)
        origin : (vdim,)

        return (coords - origin)/spacing, index of grid point
        '''
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, device=device,
                                  dtype=torch.float32, requires_grad=False)
        if len(coords.shape) == 1:
            coords = coords.unsqueeze(0)
        if not isinstance(origin, torch.Tensor):
            origin = torch.tensor(origin, device=device,
                                  dtype=torch.float32, requires_grad=False)
        if not isinstance(spacing, torch.Tensor):
            spacing = torch.tensor(spacing, device=device,
                                   dtype=torch.float32, requires_grad=False)
        idxs = (coords - origin.unsqueeze(0)) / spacing.unsqueeze(0)
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def get_coordinate(self, idxs):
        '''
        wraper
        '''
        return UniversalGrid.compute_coordinate(idxs, self._grid_spacing, self._origin, self._device)

    def get_index(self, coords):
        """
        Converts physical coordinates to grid indices, considering the grid's origin offset.

        Args:
            coords (torch.Tensor): Physical coordinates (M, vdim).
            offset (torch.Tensor, optional): Offset to apply to indices. Defaults to the
                grid's origin offset.

        Returns:
            torch.Tensor: Grid indices (M, vdim).
        """
        return UniversalGrid.compute_index(coords, self._origin, self._grid_spacing, self._device)

    def from_grid(self, grid_spacing, origin=None):
        '''
        grid : tuple of torch.Tensor
        origin: new origin, default : None --> minimum of grid
        '''
        if not isinstance(grid_spacing, torch.Tensor):
            grid_spacing = torch.tensor(grid_spacing, dtype=torch.float32, requires_grad=False)
        self._grid_spacing = grid_spacing
        if not isinstance(origin, torch.Tensor):
            origin = torch.tensor(origin, dtype=torch.float32, requires_grad=False)
        self._origin = origin
        return self

    def coarse_grain(self, multiples):
        '''
        multiples : (vdim, )
        '''
        if not isinstance(multiples, torch.Tensor):
            multiples = torch.tensor(multiples, dtype=torch.float32, requires_grad=False)
        if len(self._grid_spacing) != len(multiples):
            raise ValueError('In compatible shape of multiples, and grid spacing, {} incompatible to {}'.format(multiples.shape, self._grid_spacing.shape))
        return UniversalGrid().from_grid(self._grid_spacing * multiples, self._origin)


class LocalGrid:
    """
    A class for building a local grid to compute offsets to origin in terms of index,
    and shapes in unit of number of grid spacing, for a series of steps.
    """

    def __init__(self, origin, grid_spacing, n_sigma, device='cpu'):
        """
        Initialize the grid.

        Args:
            origin (vdim,) (float): Origin for grid in each dimension.
            grid_spacing (vdim,) (float): Spacing for grid in each dimension.
            n_sigma (vdim,), (int): Number of sigmas to include in bounds.
        """
        self._origin = make_tensor(origin, device=device)
        self._origin = self._origin.clone().detach()
        self._grid_spacing = make_tensor(grid_spacing, device=device)
        self._grid_spacing = self._grid_spacing.clone().detach()
        self._n_sigma = make_tensor(n_sigma, device=device)
        self._n_sigma = self._n_sigma.clone().detach()
        self._device = device

    @staticmethod
    def compute_bounds_X0X1(X0X1, Sigma, n_sigma):
        '''
        X0X1 : (N, vdim, 2)
        Sigma : (N, vdim)
        n_sigma: (vdim, )
        return : (N, vdim, 2)
        '''
        n_sigma = make_tensor(n_sigma, dtype=torch.float32, device=Sigma.device)
        offset = (n_sigma.unsqueeze(0) * Sigma) # (N, vdim)
        min_limits = torch.min(X0X1, dim=2).values - offset # torch.min(shape(N,vdim,2)) --> shape(N, vdim)
        max_limits = torch.max(X0X1, dim=2).values + offset
        return torch.stack([min_limits, max_limits], dim=2)

    @staticmethod
    def stack_X0X1(X0, X1):
        '''
        X0: (N, vdim)
        X1: (N, vdim)
        return (N, vdim, 2)
        '''
        return torch.stack((X0, X1), dim=2)

    @staticmethod
    def compute_bounds_X0_X1(X0, X1, Sigma, n_sigma):
        '''
        X0: (N, vdim)
        X1: (N, vdim)
        Sigma: (N, vdim)
        n_sigma: (vdim, )
        return (N, vdim, 2) float
        '''
        combined = LocalGrid.stack_X0X1(X0, X1)
        bounds = LocalGrid.compute_bounds_X0X1(combined, Sigma, n_sigma)
        return bounds

    @staticmethod
    def reduce_to_universal(shape):
        """
        Compute a universal shape across all steps using reductions.

        Args:
            shapes (torch.Tensor): Shapes of the charge boxes (N, vdim).

        Returns:
            Universal shape. (vdim, )
        """
        universal_max = torch.max(shape, dim=0).values

        return universal_max

    @staticmethod
    def compute_charge_box(X0, X1, Sigma, n_sigma, origin, grid_spacing, compare_key='index'):
        '''
        offset : (N, vdim)
        shape : (vdim,)
        '''
        if not isinstance(X0, torch.Tensor):
            raise ValueError('X0 must be a torch.Tensor')
        device = X0.device
        extremes = LocalGrid.compute_bounds_X0_X1(X0, X1, Sigma, n_sigma) # (N, vdim, 2)
        min_limit = extremes[:, :, 0] # (N, vdim)
        max_limit = extremes[:, :, 1] # (N, vdim)
        if compare_key == 'index':
            min_limit = UniversalGrid.compute_index(min_limit, origin, grid_spacing, device=device)
            max_limit = UniversalGrid.compute_index(max_limit, origin, grid_spacing, device=device)
            offset = min_limit
            shape = max_limit - min_limit + 1
            shape = LocalGrid.reduce_to_universal(shape)
            shape = shape.to(torch.int32)
        elif compare_key == 'coordinate':
            raise NotImplementedError('Not support comparation by coordinate')
            '''
            The index computation cast the coordinate to lower bound.
            I am not sure how to reconcile cases below:
            1) Minimum close to lower bound and maximum close to upper bound.
            2) Minimum close to upper bound and maximum close to lower bound.
            The shape given by the two are off by one.
            '''
        else:
            raise NotImplementedError('Only support comparation by index and coordinate')
        return offset, shape

    def compute_bounds(self, X0, X1, Sigma):
        """
        Compute the adjusted bounds for the grid.

        Args:
            X0 (torch.Tensor): Starting points of the steps (N, vdim).
            X1 (torch.Tensor): Ending points of the steps (N, vdim).
            Sigmas (torch.Tensor): Gaussian diffusion widths (N, vdim).

        Returns:
            tuple: Adjusted min and max bounds for the grid (min_limits, max_limits).
        """
        # Stack X0 and X1 to find global min and max
        bounds = LocalGrid.compute_bounds_X0_X1(X0, X1, Sigma, self._n_sigma)
        return bounds

    def compute_index(self, coords):
        """
        Compute grid indices directly from physical coordinates.

        Args:
            coords (torch.Tensor): Physical coordinates (N, vdim).
            grid_min (torch.Tensor): Minimum bounds of the grid, (vdim, ).

        Returns:
            torch.Tensor: Grid indices (N, vdim).
        """
        return UniversalGrid.compute_index(coords, self._origin, self._grid_spacing, self._device)

    def get_charge_box(self, X0, X1, Sigma, compare_key='index'):
        """
        Compute the offset and shape of the charge box for a series of steps.

        Args:
            X0 (torch.Tensor): Starting points of the steps (N, vdim).
            X1 (torch.Tensor): Ending points of the steps (N, vdim).
            Sigmas (torch.Tensor): Gaussian diffusion widths (N, vdim).

        Returns:
            dict: Offsets and shapes of the charge boxes.
        """
        return LocalGrid.compute_charge_box(X0, X1, Sigma, self._n_sigma, self._origin, self._grid_spacing, compare_key)

class QModel():
    def __new__(cls, *args, **kwargs):
        raise TypeError("This class cannot be instantiated.")

    @staticmethod
    def compute_shape_2D(X0, X1, Sigma, n_sigma, tilt_min, tilt_max):
        """
        Compute the upper and lower bounds of the parallelepiped based on tilt and sigma.
        X0, X1: Tensors defining bounding positions in the X/Y plane. (N, 2), float
        Sigma: (N, 2)
        n_sigma: (2, )
        tilt_min: Scalar tensor indicating tilt value (>0, <0, or 0).
        tilt_max: Scalar
        N_sigma: Scalar tensor for diffusion approximation. (N, 2), float
        return:
        ref : (N, 2) : bottom-left corner; values can be negative
        width : (N, )
        height : (N, )
        tilt : (N, )
        """
        if not isinstance(X0, torch.Tensor):
            raise ValueError('X0 must be a torch.Tensor')
        device = X0.device
        tilt = (X1[:,1]-X0[:,1])/(X1[:,0]-X0[:,0]) # (N,); inf can be in the result
        n_sigma = make_tensor(n_sigma, device=device)

        mask = (torch.abs(tilt) < tilt_max) & (torch.abs(tilt) >= tilt_min)
        positive_mask = mask & (tilt >= 0)
        negative_mask = mask & (tilt < 0)
        N_sigma = n_sigma[None,:] * Sigma
        nsx, nsy = [N_sigma[:,i] for i in range(2)] # (N, ), (N, )

        minX0X1 = torch.min(X0, X1) # (N, 2); xmin, ymin; tilt > 0 , matched x, y; tilt < 0, mismatched x, y, i.e., (X0[0], X1[1]) or (X0[1], X1[0])
        maxX0X1 = torch.max(X0, X1) # (N, 2);
        width = maxX0X1[:,0] - minX0X1[:,0] + 2 * nsx# (N, )

        dy = 2 * nsx * torch.abs(tilt) # (N,)

        height = torch.where(
            mask,
            dy + 2 * nsy,
            maxX0X1[:,1] - minX0X1[:,1] + 2 * nsy
        )

        ref = torch.where(
            positive_mask[:,None],
            minX0X1 - N_sigma - torch.stack([torch.zeros_like(dy), dy], dim=1),
            torch.stack(
                [minX0X1[:,0] - nsx,
                 maxX0X1[:,1] - nsy], dim=1
            )
        )
        if torch.sum(~mask).item() > 0:
            ref[~mask] = (minX0X1 - N_sigma)[~mask]

        return ref, width, height, tilt

    @staticmethod
    def compute_shape_2D_stacked(X0, X1, Sigma, n_sigma, tilt_min, tilt_max):
        ref, width, height, tilt = QModel.compute_shape_2D(X0, X1, Sigma, n_sigma, tilt_min, tilt_max)
        return torch.stack([ref[:,0], ref[:,1], width, height, tilt], dim=1)

    @staticmethod
    def mask2D(X0, X1, Sigma, n_sigma, tilt_min, tilt_max, origin, grid_spacing, box_offset, box_shape):
        '''
        X0, X1: Tensors defining start/end positions. (N, 2), float
        Sigma: (N, 2)
        n_sigma: (2, )
        tilt_min: axis1 / axis0, Scalar tensor indicating tilt value (>0, <0, or 0).
        tilt_max: Scalar
        N_sigma: Scalar tensor for diffusion approximation. (N, 2), float
        origin : (2,) float
        grid_spacing : (2,) float
        box_offset : (N,2), int
        box_shape : (2,), int
        '''
        # tilt is height / width
        # width maps to x, axis 0
        # height maps to y, axis 1
        if not isinstance(X0, torch.Tensor):
            raise ValueError('X0 must be a torch')
        device = X0.device

        ref, width, height, tilt = QModel.compute_shape_2D(X0, X1, Sigma, n_sigma, tilt_min, tilt_max)
        start_x, start_y = ref[:,0], ref[:,1]

        # historical swap...
        cols = torch.arange(box_shape[0], dtype=torch.int32, device=device).unsqueeze(1) # (box_width, 1)
        rows = torch.arange(box_shape[1], dtype=torch.int32, device=device).unsqueeze(0) # (1, box_height)

        start = torch.stack([start_x, start_y], dim=1) # (N, 2)
        local_startidx = UniversalGrid.compute_index(start, origin, grid_spacing, device=device) - box_offset # (N, 2)

        tilt_mask = (torch.abs(tilt) < tilt_max) & (torch.abs(tilt) >= tilt_min)
        row_offsets_per_col = torch.where(
            tilt_mask[:, None, None],
            torch.clamp((cols.unsqueeze(0) - local_startidx[:,0][:, None, None]) * tilt[:, None, None] , min=0), # shape (N, box_width, 1)
            0
        )
        row_starts_per_col = local_startidx[:,1][:, None, None] + row_offsets_per_col # shape (N, box_width, 1)
        row_ends_per_col = row_starts_per_col + torch.ceil(height[:, None, None]/grid_spacing[1]) + 1 # shape (N, box_width, 1)
        cols = cols.unsqueeze(0)
        rows = rows.unsqueeze(0)
        col_mask = (
            (cols >= local_startidx[:,0][:,None,None])
            & (cols < (local_startidx[:,0] + torch.floor(width/grid_spacing[0]))[:,None,None] + 1)
        ) # (N, box_width, 1)
        mask = (
            (rows >= row_starts_per_col) & (rows < row_ends_per_col) & col_mask
        )
        return mask

    def compute_shape_3D(X0, X1, Sigma, n_sigma, ax=(0,1,2), tilt_min=(1E-3,1E-3,1E-3), tilt_max=(1E3,1E3,1E3)):
        '''
        ax : tuple (3, )
        tilt : ax[1]/ax[0], ax[2]/ax[0], ax[2]/ax[1]
        '''
        if not isinstance(X0, torch.Tensor):
            raise ValueError('X0 must be a torch')

        device = X0.device
        tilt = torch.stack(
            [ (X1[:,ax[j]]-X0[:,ax[j]])/(X1[:,ax[i]]-X0[:,ax[i]])
              for i, j in zip([0, 0, 1], [1, 2, 2])]
        ) # tilt[3] will be recomputed according to the shape; as projection must be larger than the original area
        n_sigma = make_tensor(n_sigma, device=device)
        tilt_min = make_tensor(tilt_min, device=device)
        tilt_max = make_tensor(tilt_max, device=device)
        N_sigma = n_sigma[None,:] * Sigma

        condition = X0[:, ax[0]] < X1[:, ax[0]]  # Shape: (N,)
        X0new = torch.where(condition.unsqueeze(1), X0, X1)
        X1new = torch.where(condition.unsqueeze(1), X1, X0)
        mask12 = ( # ax1/ax0, ax2/ax0
            (torch.abs(tilt[0]) < tilt_max[0])  & (torch.abs(tilt[0]) >= tilt_min[0])
            & (torch.abs(tilt[1]) < tilt_max[1])  & (torch.abs(tilt[1]) >= tilt_min[1])
        )
        # for a reasonable tilt[1], tilt[2]
        # dax1 = tilt[1] * N_sigma[:,ax[0]] * 2 # (N, )
        # dax2 = tilt[2] * N_sigma[:,ax[0]] * 2 # (N, )
        dax1 = tilt[0] * N_sigma[:,ax[0]] * 2 # (N, )
        dax2 = tilt[1] * N_sigma[:,ax[0]] * 2 # (N, )
        X0new_ax12 = torch.stack([X0new[:, ax[1]], X0new[:, ax[2]]], dim=1)
        X0new_shift_ax12 = X0new_ax12 - torch.stack([dax1, dax2], dim=1)
        Sigma_ax12 = torch.stack([Sigma[:, ax[1]], Sigma[:, ax[2]]], dim=1)
        n_sigma_ax12 = torch.tensor([n_sigma[ax[1]], n_sigma[ax[2]]])

        X0_ax12 = torch.stack([X0[:, ax[1]], X0[:, ax[2]]], dim=1)
        X1_ax12 = torch.stack([X1[:, ax[1]], X1[:, ax[2]]], dim=1)

        stacked_2Dshape = torch.where(
            mask12[:, None],
            QModel.compute_shape_2D_stacked(X0new_shift_ax12, X0new_ax12, Sigma_ax12, n_sigma_ax12, tilt_min[2], tilt_max[2]), # (N, 5)
            QModel.compute_shape_2D_stacked(X0_ax12, X1_ax12, Sigma_ax12, n_sigma_ax12, tilt_min[2], tilt_max[2]) # degenerated (N, 5)
        )
        ref_ax12 = stacked_2Dshape[:,:2]
        width_ax12 = stacked_2Dshape[:,2]
        height_ax12 = stacked_2Dshape[:,3]
        tilt_ax12 = stacked_2Dshape[:,4]

        ref_ax0 = X0new[:,ax[0]] - N_sigma[:,ax[0]]
        depth_ax0 = X1new[:,ax[0]]-X0new[:,ax[0]] + 2 * N_sigma[:,ax[0]]
        tilt_ax01 = tilt[0]
        tilt_ax02 = tilt[1]
        # further dealing with degeneracy for tilt_ax00, tilt_ax01 necessary
        # missing tilt ax 12 degenerated?

        return ref_ax0, ref_ax12, depth_ax0, width_ax12, height_ax12, tilt_ax01, tilt_ax02, tilt_ax12

    @staticmethod
    def mask3D(X0, X1, Sigma, n_sigma, tilt_min, tilt_max, origin, grid_spacing, box_offset, box_shape, **kwargs):
        '''
        X0, X1: Tensors defining start/end positions. (N, 3), float
        Sigma: (N, 3)
        n_sigma: (3, )
        tilt_min: min of ax[1]/ax[0], ax[2]/ax[0], ax[2]/ax[1], tuple
        tilt_max: min of ax[1]/ax[0], ax[2]/ax[0], ax[2]/ax[1], tuple
        origin : (3,) float
        grid_spacing : (3,) float
        box_offset : (N, 3), int, in grid indexing
        box_shape : (3,), int, range of array indexing

        kwargs['ax'] = (0, 1, 2) or others
        '''
        if not isinstance(X0, torch.Tensor):
            raise ValueError('X0 must be a torch')

        device = X0.device
        # tilt is height / width
        # width maps to x, axis 0
        # height maps to y, axis 1
        ax = kwargs.get('ax', (0, 1, 2))
        # ax0, ax[0], is just from min to max
        # ax1, ax[1], is controlled by ax0
        # ax2, ax[2], is controlled by ax0 and ax1
        idxax0 = torch.arange(box_shape[ax[0]], dtype=torch.int32, device=device).unsqueeze(1).unsqueeze(2) # (box_depth, 1, 1)
        idxax1 = torch.arange(box_shape[ax[1]], dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(2) # (1, box_width, 1)
        idxax2 = torch.arange(box_shape[ax[2]], dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(1) # (1, 1, box_height)

        tilt_min_ax01 = make_tensor(tilt_min[0], device=device)
        tilt_min_ax02 = make_tensor(tilt_min[1], device=device)
        tilt_min_ax12 = make_tensor(tilt_min[2], device=device)

        tilt_max_ax01 = make_tensor(tilt_max[0], device=device)
        tilt_max_ax02 = make_tensor(tilt_max[1], device=device)
        tilt_max_ax12 = make_tensor(tilt_max[2], device=device)

        ref_ax0, ref_ax12, depth_ax0, width_ax12, height_ax12, tilt_ax01, tilt_ax02, tilt_ax12 = QModel.compute_shape_3D(X0, X1, Sigma, n_sigma, ax, tilt_min, tilt_max)
        start = [ref_ax0, ref_ax12[:,0], ref_ax12[:,1]] # (N, 3)
        start = torch.stack([x for _, x in sorted(zip(ax, start))], dim=1)

        local_array_startidx = UniversalGrid.compute_index(start, origin, grid_spacing, device=device) - box_offset # (N, 3) # array indices

        tilt_mask12 = ( # ax1/ax0, ax2/ax0
            (torch.abs(tilt_ax01) < tilt_max_ax01)  & (torch.abs(tilt_ax01) >= tilt_min_ax01)
            & (torch.abs(tilt_ax02) < tilt_max_ax02)  & (torch.abs(tilt_ax02) >= tilt_min_ax02)
        ) # (N, )
        tilt_mask_ax12 = (
            (torch.abs(tilt_ax12) < tilt_max_ax12) & (torch.abs(tilt_ax12) >= tilt_min_ax12)
        )

        # dax0 * tilt01 = dax1
        # dax0 * tilt01 = dax2
        # dax1 * tilt12 = dax2

        idxax1_offset_per_ax0 = torch.where(tilt_mask12[:, None, None, None],  # (N, 1, 1, 1)
                                            torch.clamp(tilt_ax01[:,None, None, None] *  (idxax0.unsqueeze(0) - local_array_startidx[:,ax[0]][:, None, None, None]),
                                                        min=0) # (N, 1, 1, 1) * {(1, box_depth, 1, 1) - (N, 1, 1, 1)} --> (N, box_depth, 1, 1)
                                            , 0) # (N, box_depth, 1, 1)

        idxax2_offset_per_ax0_ax1 = torch.where(tilt_mask_ax12[:, None, None, None], # (N, 1, 1, 1)
                                                torch.clamp(
                                                    tilt_ax12[:,None, None, None] * (idxax1.unsqueeze(0) - local_array_startidx[:,ax[1]][:, None, None, None]) # (N, 1, 1, 1) * {(1, 1, box_width, 1) - (N, 1, 1, 1)} --> (N, 1, box_width, 1)
                                                    , min=0), 0) # --> (N, box_depth, box_width, 1)

        idxax1_start_per_ax0 = local_array_startidx[:,ax[1]][:, None, None, None] + idxax1_offset_per_ax0 # (N, 1, 1, 1) + (N, box_depth, 1, 1) --> (N, box_depth, 1, 1)
        idxax1_start_per_ax0 = idxax1_start_per_ax0.clamp(min=0, max=box_shape[ax[1]]-1) # (N, box_depth, 1, 1)

        idxax2_start_per_ax0_ax1 = local_array_startidx[:,ax[2]][:, None, None, None] + idxax2_offset_per_ax0_ax1 # (N, 1, 1, 1) + (N, box_depth, box_width, 1) --> (N, box_depth, box_width, 1)
        idxax2_start_per_ax0_ax1 = idxax2_start_per_ax0_ax1.clamp(min=0, max=box_shape[ax[2]]-1) # (N, box_depth, box_width, 1)
        idxax2_start_per_ax0_ax1 = torch.floor(idxax2_start_per_ax0_ax1)

        idxax1_end_per_ax0 = idxax1_start_per_ax0 + torch.ceil(width_ax12[:, None, None, None]/grid_spacing[ax[1]]) + 1 + 1 # (N, box_depth, 1, 1) + (N, 1, 1, 1) --> (N, box_depth, 1, 1)
        idxax2_end_per_ax0_ax1 = idxax2_start_per_ax0_ax1 + torch.ceil(height_ax12[:, None, None, None]/grid_spacing[ax[2]]) + 1 + 1# ((N, box_depth, box_width, 1) + (N, 1, 1, 1) --> (N, box_depth, box_width, 1)

        # TBD; another reference
        (_, ref_ax21, _,
         width_ax21, height_ax21, _, _,
         _) = QModel.compute_shape_3D(X0, X1, Sigma, n_sigma,
                                      (ax[0], ax[2], ax[1]),
                                      (tilt_min[0], tilt_min[2], tilt_min[1]),
                                      (tilt_max[0], tilt_max[2], tilt_max[1]))

        idxax2_offset_per_ax0 = torch.where(
            tilt_mask12[:, None, None, None],
            torch.clamp(tilt_ax02[:,None, None, None] * (idxax0.unsqueeze(0) - local_array_startidx[:,ax[0]][:, None, None, None]) # (N, 1, 1, 1) * {(1, box_depth, 1, 1) - (N, 1, 1, 1)} --> (N, box_depth, 1, 1)
        , min=0),
        0
        ) # (N, box_depth, 1, 1)

        # TBD; another reference
        start_ax21 = [ref_ax0, ref_ax21[:,0], ref_ax21[:,1]] # (N, 3)
        start_ax21 = torch.stack([x for _, x in sorted(zip([ax[0], ax[2], ax[1]], start_ax21))], dim=1)
        local_array_startidx_ax21 = UniversalGrid.compute_index(start_ax21, origin, grid_spacing, device=device) - box_offset # (N, 3) # array indices

        idxax2_start_per_ax0 = local_array_startidx_ax21[:, ax[2]][:,None,None,None] + idxax2_offset_per_ax0
        idxax2_start_per_ax0 = idxax2_start_per_ax0.clamp(min=0, max=box_shape[ax[2]]-1) # (N, box_depth, 1, 1)
        idxax2_end_per_ax0 = idxax2_start_per_ax0 + torch.ceil(width_ax21[:, None, None, None]/grid_spacing[ax[2]]) + 1 +1 # (N, box_depth, 1, box_depth)

        idxax0 = idxax0.unsqueeze(0)
        idxax1 = idxax1.unsqueeze(0)
        idxax2 = idxax2.unsqueeze(0)

        n_sigma = make_tensor(n_sigma, device=device)

        minX0X1 = torch.min(X0, X1) - n_sigma.unsqueeze(0) * Sigma
        maxX0X1 = torch.max(X0, X1) + n_sigma.unsqueeze(0) * Sigma
        idxlw = UniversalGrid.compute_index(minX0X1, origin, grid_spacing, device=device) - box_offset
        idxup = UniversalGrid.compute_index(maxX0X1, origin, grid_spacing, device=device) - box_offset + 1

        idxax0_mask = (
            (idxax0 >= local_array_startidx[:,ax[0]][:, None, None, None]) # (N, box_depth, 1, 1)
            & (idxax0 < (torch.ceil(depth_ax0/grid_spacing[ax[0]]) + local_array_startidx[:,ax[0]] + 2)[:, None, None, None]) # (N, box_depth, 1, 1)

        )
        idxax1_mask = (
            (idxax1 >= torch.floor(idxax1_start_per_ax0)) # (1, 1, box_width, 1) >= (N, box_depth, 1, 1) --> (N, box_depth, box_width, 1)
            & (idxax1 < idxax1_end_per_ax0 + 1) # (1, 1, box_width, 1) < (N, box_depth, 1, 1) --> (N, box_depth, box_width, 1)
            & (idxax1 >= idxlw[:,1][:, None, None, None])
            & (idxax1 < idxup[:,1][:, None, None, None])
        )
        idxax2_mask = (
            (idxax2 >= idxax2_start_per_ax0_ax1) # (1, 1, box_height) >= (N, box_depth, box_width, 1) --> (N, box_depth, box_width, box_height)
            & (idxax2 < idxax2_end_per_ax0_ax1 + 1) # (1, 1, box_height) >= (N, box_depth, box_width, 1) --> (N, box_depth, box_width, box_height)
            & (idxax2 >= idxax2_start_per_ax0)
            & (idxax2 < idxax2_end_per_ax0 + 1)
            & (idxax2 >= idxlw[:,2][:, None, None, None])
            & (idxax2 < idxup[:,2][:, None, None, None])
        )
        mask = (
            idxax0_mask & idxax1_mask & idxax2_mask
        )
        return mask


    # @torch.jit.script
    @staticmethod
    def GaussConvLine3D(Q, X0, X1, Sigma, x, y, z):
        '''
        Q (N,)
        X0 (N, 3)
        X1 (N, 3)
        Sigma (N, 3)
        x (N, other shape)
        y (N, other shape)
        z (N, other shape)
        '''
        sqrt2 = 1.4142135623730951

        # prepare for brodcasting
        num_dims_to_add = x.ndimension() - 1
        shape_new = (x.shape[0],) + (1,)* num_dims_to_add


        x0, y0, z0 = [X0[:,i].view(shape_new) for i in range(3)]
        x1, y1, z1 = [X1[:,i].view(shape_new) for i in range(3)]
        sx, sy, sz = [Sigma[:,i].view(shape_new) for i in range(3)]

        dx01 = x0 - x1 # (N,)
        dy01 = y0 - y1 # (N,)
        dz01 = z0 - z1 # (N,)
        sxsy2 = (sx*sy)**2 # (N,)
        sxsz2 = (sx*sz)**2 # (N,)
        sysz2 = (sy*sz)**2 # (N,)

        sx2 = sx**2 # (N,)
        sy2 = sy**2 # (N,)
        sz2 = sz**2 # (N,)

        deltaSquare = (
            sysz2 * dx01**2
            + sxsy2 * dz01**2 + sxsz2 * dy01**2
        ) # (N,)

        deltaSquareSqrt = torch.sqrt(deltaSquare) # (N,)

        QoverDeltaSquareSqrt4pi = Q.view(shape_new) / deltaSquareSqrt / 4 / np.pi # (N,)
        erfArgDenominator = sqrt2 * deltaSquareSqrt * sx * sy * sz # (N,)

        charge = ((-QoverDeltaSquareSqrt4pi * torch.exp(
            -sy2 * torch.pow(x * dz01 + (z1*x0 - z0*x1) - z * dx01, 2) *0.5/deltaSquare ))
                  * torch.exp(
                      -sx2 * torch.pow(y * dz01 + (z1*y0 - z0*y1) - z * dy01, 2) *0.5/deltaSquare)
                  * torch.exp(
                      -sz2 * torch.pow(y * dx01 + (x1*y0 - x0*y1) - x * dy01, 2) *0.5/deltaSquare)) * (
                          torch.erf((
                              sysz2 * (x - x0) * dx01 +
                              sxsy2 * (z - z0) * dz01 +
                              sxsz2 * (y - y0) * dy01
                          )/erfArgDenominator) -
                          torch.erf((
                              sysz2 * (x - x1) * dx01 +
                              sxsy2 * (z - z1) * dz01 +
                              sxsz2 * (y - y1) * dy01
                          )/erfArgDenominator)
                      )

        return charge

class QEff3D():
    '''QEff'''
    def __init__(self, origin, grid_spacing, offset, shape,
                 method='gauss_legendre_4_4_4',
                 device='cpu'
                 ):
        '''xspace, yspcae, tspace are tuples of three numbers
        defining the range and number of grid points along one dimension
        of the charge space.
        Note: The end point is always included.
        Input (1, 2, 3) will yield ([1, 1.5, 2].
        '''
        # bad as it needs to be (N, other shapes)
        # TBD
        # self._grid = tuple(
        #     torch.arange(o+off*step, o+(off+shp)*step+0.1*step,step) for o, step, off, shp, in zip(origin, grid_spacing, offset, shape)
        # )

        self._box_shape = shape
        self._box_offset = offset if isinstance(offset, torch.Tensor) \
            else torch.tensor(offset, dtype=torch.int64, requires_grad=False, device=device)
        self._origin = origin
        self._grid_spacing = grid_spacing

        self._method, self._npoints = QEff3D.parse_method(method)

        self._device = device

    @property
    def origin(self):
        return self._origin
    @property
    def grid_spacing(self):
        return self._grid_spacing
    @property
    def box_shape(self):
        return self._box_shape
    @property
    def box_offset(self):
        return self._box_offset

    @staticmethod
    def parse_method(method):
        if method == 'cube_corner':
            raise NotImplementedError("cube_corner not implemented.")
        elif 'gauss_legendre' in method[0:len('gauss_legendre')]:
            npoints = method[len('gauss_legendre')+1:].split('_')
            npoints = tuple(int(p) for p in npoints)
            return 'gauss_legendre', npoints
        else:
            msg = (
                f"Method {method} not supported. "
                "Available options are cube_corner, gauss_legendre_n_n_n, "
                "where n means n-point Gauss-Legendre Quadrature. "
                "For instance, gauss_legendre_2 means two-point "
                "Gauss-Legendre Quadrature."
            )
            raise NotImplementedError(msg)

    @staticmethod
    def create_w1d_GL(npt, spacing, device='cpu'):
        '''
        npt : integer
        spacing : float
        return a tensor of weights of n-point GL quadrature after correcting for length of intervals
        '''
        _, weights = sp.special.roots_legendre(npt)
        w1d = torch.tensor(weights, dtype=torch.float32, requires_grad=False, device=device) * spacing/2
        return w1d

    @staticmethod
    def create_w1ds(method, npoints, grid_spacing, device='cpu'):
        '''
        method : str
        npoints : (3, ), integers
        grid_spacing : (3, ), float
        return (3, ) list with each element is a tensor of weights of n-point GL quadrature after correcting for length of intervals
        '''
        if method != 'gauss_legendre':
            raise NotImplementedError('Not implemented method but gauss legendre quadrature')
        w1ds = [None, None, None]
        for ipt, npt in enumerate(npoints):
            w1ds[ipt] = QEff3D.create_w1d_GL(npt, grid_spacing[ipt], device)
        return w1ds

    @staticmethod
    def create_weight_block(w1ds):
        '''create a weight block in 3D'''
        w3d = w1ds[0][:, None, None] * w1ds[1][None, :, None] * w1ds[2][None, None, :]
        return w3d

    @staticmethod
    def create_u1d_GL(npt, device='cpu'):
        '''
        npt : integer
        return: a tensor of coefficients for interpolations at roots of npt-order GL polynomials
        '''
        roots, _ = sp.special.roots_legendre(npt)
        roots = torch.tensor(roots, dtype=torch.float32, requires_grad=False, device=device)
        u = (roots+1)/2
        u1d = torch.empty([npt, 2], dtype=torch.float32, requires_grad=False, device=device)
        u1d[:, 0] = 1-u
        u1d[:, 1] = u
        return u1d

    @staticmethod
    def create_u1ds(method, npoints, device='cpu'):
        '''
        method : str
        npoints : (3, ) integers
        return (3, ) list with each elment is a tensor of coefficients for interpolations
        '''
        u1ds = []
        for ipt, npt in enumerate(npoints):
            u1ds.append(QEff3D.create_u1d_GL(npt, device))
        return u1ds

    @staticmethod
    def create_u_block(u1ds):
        '''create a weight block for u in 3D
        Requirements: u1d with a shape of (npt, 2) where npt means n-point GL quad rule.
        '''
        for i in range(3):
            if u1ds[i].shape[1] != 2:
                raise ValueError('u1d must have a shape of (npt, 2)')
        u3d = (
            u1ds[0][:, None, None, :, None, None] *
                u1ds[1][None, :, None, None, :, None] *
                u1ds[2][None, None, :, None, None, :]
        )
        return u3d

    @staticmethod
    def create_wu_block(w3d, u3d):
        '''
        w3d (L,M,N)
        u3d (L,M,N,2,2,2)
        return w3d[i,j,k] * u3d{i,j,k,:,:,:]
        '''
        return w3d[:,:,:,None,None,None] * u3d

    @staticmethod
    def create_sampling_1d_GL(origin_1d, grid_spacing_1d, offset_1d, shp_1d, npt, device='cpu'):
        '''
        origin_1d : float
        grid_spacing_1d : float
        offset_1d : (Nsteps, ) tensor of integer
        shp_1d : integer, I/J/K where I/J/K is the number of grid points along x/y/z
        npt : npoints GL quad rule
        return: samplings (Nsteps, L/M/N, I/J/K-1).
        '''
        shp_idx_1d = torch.arange(shp_1d+1, dtype=torch.float32, requires_grad=False, device=device) # (I/J/K, )
        idx_2d = offset_1d[:,None] + shp_idx_1d[None, :]
        corners_1d = origin_1d + idx_2d * grid_spacing_1d # (Nsteps, I/J/K)
        roots, _ = sp.special.roots_legendre(npt) # (L/M/N, )
        roots = torch.tensor(roots, dtype=torch.float32, requires_grad=False, device=device)
        half_delta = (corners_1d[:, 1:] - corners_1d[:, :-1])/2. # (Nsteps, I/J/K-1)
        avg = (corners_1d[:,1:] + corners_1d[:,:-1])/2. # (Nsteps, I/J/K)
        sampling_1d = half_delta[:, None, :] * roots[None, :, None] + avg[:, None, :] # (Nsteps, L/M/N, I/J/K-1)
        return sampling_1d

    @staticmethod
    def create_sampling_1ds(method, npoints, origin, grid_spacing, offset, shape, device='cpu'):
        '''
        method : str
        origin : tuple of three floats
        grid_spaing : tuple of three floats
        offset : (Nsteps, 3) for three dimensions
        shape : (3, ) for three dimensions
        npoints : (3, ) for GL quad rule
        return: list of sampling points with each element is a tensor with a shape of (Nsteps, L/M/N, I/J/K-1)
        '''
        sampling_1ds = []
        if method != 'gauss_legendre':
            raise NotImplementedError('Not implemented method but gauss legendre quadrature')
        for i in range(3):
            sampling_1ds.append(QEff3D.create_sampling_1d_GL(origin[i], grid_spacing[i],
                                                             offset[:,i], shape[i], npoints[i], device))
        return sampling_1ds

    @staticmethod
    def eval_line_conv_gaus(Q, X0, X1, Sigma, x, y, z):
        '''
        Q (Nsteps, ),
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        x, y, z are in a shape of (Nsteps, L/M/N, I/J/K-1)
        '''
        charge = QModel.GaussConvLine3D(
            Q, X0, X1, Sigma,
            x[:, :, None, None, :, None, None], # (Nsteps, L, 1, 1, I-1, 1, 1)
                        y[:, None, :, None, None, :, None], # (Nsteps, 1, M, 1, 1, J-1, 1)
                        z[:, None, None, :, None, None, :]) # (Nsteps, 1, 1, N, 1, 1, K-1)
        return charge

    @staticmethod
    def eval_qmodel(Q, X0, X1, Sigma, x, y, z, **kwargs):
        '''
        Q (Nsteps, )
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        x, y, z are in a shape of (Nsteps, L/M/N, I/J/K-1)
        '''
        qmodel = kwargs.get('qmodel', None)
        if qmodel is None:
            qmodel = QModel.GaussConvLine3D

        charge = qmodel(Q, X0, X1, Sigma, x[:, :, None, None, :, None, None], # (Nsteps, L, 1, 1, I-1, 1, 1)
                        y[:, None, :, None, None, :, None], # (Nsteps, 1, M, 1, 1, J-1, 1)
                        z[:, None, None, :, None, None, :]) # (Nsteps, 1, 1, N, 1, 1, K-1)

        return charge

    @staticmethod
    def eval_mask(Q, X0, X1, Sigma, offset, shape, origin, grid_spacing, n_sigma):
        '''
        '''
        if not isinstance(X0, torch.Tensor):
            raise ValueError('X0 must be a torch.Tensor')
        device = X0.device

        box_shape = make_tensor(shape, dtype=torch.int32, device=device)
        box_offset = make_tensor(offset, dtype=torch.int64, device=device)
        tilt_min = 2./box_shape.float()
        tilt_max = box_shape.float()/2.
        m = QModel.mask3D(X0, X1, Sigma, n_sigma, tilt_min, tilt_max, origin, grid_spacing, box_offset, box_shape, ax=(0,1,2)) # (N, I, J, K)
        # Expand the dimensions of the tensor to (1, 1, 1, I, J, K)
        expanded_m = m.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        return expanded_m

    @staticmethod
    def eval_qeff(Q, X0, X1, Sigma, offset, shape, origin, grid_spacing, method, npoints, **kwargs):
        '''
        '''
        if not isinstance(X0, torch.Tensor):
            raise ValueError('X0 must be a torch.Tensor')
        device = X0.device

        usemask = kwargs.get('usemask', False)
        n_sigma = kwargs.get('n_sigma', False)
        if usemask and not n_sigma:
            raise ValueError('n_sigma must be given when using masks')

        quaddim = kwargs.get('convdim', (0,1,2)) # not used yet; place holder for future extension

        if quaddim:
            usequad = True
        else:
            usequad = False

        u1ds = QEff3D.create_u1ds(method, npoints, device)
        w1ds = QEff3D.create_w1ds(method, npoints, grid_spacing, device)
        ublock = QEff3D.create_u_block(u1ds)
        wblock = QEff3D.create_weight_block(w1ds)

        x, y, z = QEff3D.create_sampling_1ds(method, npoints, origin, grid_spacing, offset, shape, device)

        lmn = tuple(wblock.shape)
        lmn_prod = np.prod(lmn, dtype=int)
        rst = ublock.shape[3:]

        kernel = QEff3D.create_wu_block(wblock, ublock)
        kernel = torch.flip(kernel, [3, 4, 5]) # it does not matter we flip at first or we multiply w and u at first
        kernel = kernel.view(lmn_prod, 1, rst[0], rst[1], rst[2]) # out_channel, in_channel/groups, R, S, T

        # if usemask:
        #     box_shape = make_tensor(shape, dtype=torch.int32, device=device)
        #     box_offset = make_tensor(offset, dtype=torch.int64, device=device)
        #     tilt_min = 2./box_shape.float()
        #     tilt_max = box_shape.float()/2.
        #     m = QModel.mask3D(X0, X1, Sigma, n_sigma, tilt_min, tilt_max, origin, grid_spacing, box_offset, box_shape, ax=(0,1,2)) # (N, I, J, K)
        #     L, M, N = npoints
        #     # Expand the dimensions of the tensor to (1, 1, 1, I, J, K)
        #     expanded_m = m.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        #     # Tile the tensor to repeat it along the new dimensions
        #     # repeated_m = m.repeat(L, M, N, 1, 1, 1)
        #     # charge = torch.where(repeated_m, QEff3D.eval_qmodel(Q, X0, X1, Sigma, x, y, z, **kwargs), 0)
        #     charge = torch.where(expanded_m, QEff3D.eval_qmodel(Q, X0, X1, Sigma, x, y, z, **kwargs), 0)
        # else:
        #     charge = QEff3D.eval_qmodel(Q, X0, X1, Sigma, x, y, z, **kwargs)
        nbtensor = len(Q) * float(shape[0]) * float(shape[1]) * float(shape[2]) * lmn_prod * 4 / 1E9
        nbatch = int(nbtensor.item()) + 1
        batch_size = len(Q) // nbatch + 1
        qeff = []
        nQ = len(Q)
        # print(nQ, shape, nbtensor, shape[0] * shape[1] *shape[2], lmn_prod, nbatch, batch_size)
        zchunk = 400
        for i in range(0, nQ, batch_size):
            s = slice(i, min(i+batch_size, nQ))
            # print(s, nQ, shape, nbatch, batch_size)
            if z[s].size(2) > zchunk:
                qjs = []
                for j in range(0, z[s].size(2), zchunk):
                    qj = QEff3D.eval_qmodel(Q[s], X0[s], X1[s], Sigma[s], x[s], y[s], z[s][:,:,j:j+zchunk])
                    qjs.append(qj)
                charge = torch.cat(qjs, dim=-1)
            else:
                charge = QEff3D.eval_qmodel(Q[s], X0[s], X1[s], Sigma[s], x[s], y[s], z[s], **kwargs)

            charge = charge.view(len(Q[s]), lmn_prod, shape[0], shape[1], shape[2]) # batch, channel, D1, D2, D3
            charge = torch.nn.functional.pad(charge, pad=(rst[2]-1, rst[2]-1, rst[1]-1, rst[1]-1,
                                                rst[0]-1, rst[0]-1), mode="constant", value=0)
            charge = torch.nn.functional.conv3d(charge, kernel, padding='valid',
                                              groups=lmn_prod)

            charge = charge.view(len(Q[s]), lmn[0], lmn[1], lmn[2], shape[0]+1, shape[1]+1, shape[2]+1)
            qeff.append(torch.sum(charge, dim=[1, 2, 3]))

        return torch.cat(qeff, dim=0)

    def _create_w1ds(self):
        '''1D weights with the size (L, M, N) for x, y, z
        '''
        return QEff3D.create_w1ds(self._method, self._npoints, self._grid_spacing, self._device)

    def _create_u1ds(self):
        '''1D weights for trilinear interpolation
            with the size (L, M, N, 2, 2, 2) for x, y, z
        '''
        return QEff3D.create_u1ds(self._method, self._npoints, self._device)

    def _create_sampling_1d(self):
        '''create grid in 1d with a size of (L, I), (M, ).
        Note:
        1. A uniform spacing of cubes is assumed.
        2. (b-a)/2 of each interval is included in the weight
        '''
        sampling_1ds = QEff3D.create_sampling_1ds(self._method, self._origin, self._grid_spacing, self._box_offset, self._box_shape, self._npoints, self._device)
        return sampling_1ds

    def create_qeff_noweight(self, Q, X0, X1, Sigma, x, y, z):
        '''eval Q on sampling points without weights
        Q is scalar,
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        x, y, z are in a shape of (Nsteps, L/M/N, I/J/K-1)
        '''
        return QEff3D.eval_line_conv_gaus(Q, X0, X1, Sigma, x, y, z)

    def create_qeff(self, Q, X0, X1, Sigma, **kwargs):
        '''create Qeff multiplied by weights; output is squeezed'''
        qeff = QEff3D.eval_qeff(Q, X0, X1, Sigma, self._box_offset, self._box_shape,
                                self._origin, self._grid_spacing, self._method, self._npoints,
                                device=self._device,
                                **kwargs)
        return qeff
