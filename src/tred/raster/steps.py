import torch
from torch import Tensor

from ..types import index_dtype, MAX_INDEX, MIN_INDEX
# from ..utils import to_tensor


def to_tensor(source, device, dtype=torch.float32):
    '''Aliasing or create a tensor if not existing.
    Result tensor will be moved to the device
    '''
    if isinstance(source, torch.Tensor):
        t = source.to(dtype)
        t.requires_grad = False
        t = t.to(device)
    else:
        t = torch.tensor(source, dtype=dtype, requires_grad=False,
                         device=device)
    return t


def compute_coordinate(idxs: Tensor, origin, grid_spacing, device='cpu'):
    '''
    Arguments:
        idxs : (N, vdim); index of grid point, Tensor of index_dtype
        origin : (vdim,), grid origin
        grid_spacing : (vdim, ), grid spacing
    return
        origin + spacing * idx
    '''
    fidxs = to_tensor(idxs, device=device, dtype=torch.float32)
    assert torch.any(fidxs <= MAX_INDEX), 'Overflow of index_dtype'
    assert torch.any(fidxs >= MIN_INDEX), 'Underflow of index_dtype'
    idxs = to_tensor(idxs, device=device, dtype=index_dtype)

    if idxs.dim() == 1:
        idxs = idxs.unsqueeze(0)
    origin = to_tensor(origin, device=idxs.device, dtype=torch.float32)
    grid_spacing = to_tensor(grid_spacing, device=device, dtype=torch.float32)
    return origin.unsqueeze(0) + idxs * grid_spacing.unsqueeze(0)


def compute_index(coords, origin, grid_spacing, device='cpu'):
    '''
    Arguments:
        coords : (N, vdim)
        origin : (vdim,)
        grid_spacing : (vdim, )

    return
        (coords - origin)//spacing, index of grid point
    '''
    coords = to_tensor(coords, device)
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
    origin = to_tensor(origin, device)
    grid_spacing = to_tensor(grid_spacing, device)
    idxs = (coords - origin.unsqueeze(0)) / grid_spacing.unsqueeze(0)

    assert torch.any(idxs <= MAX_INDEX), 'Overflow of index_dtype'
    assert torch.any(idxs >= MIN_INDEX), 'Underflow of index_dtype'
    return idxs.floor().to(index_dtype)


def compute_bounds_X0X1(X0X1, Sigma, n_sigma):
    '''
        X0X1 : (N, vdim, 2)
        Sigma : (N, vdim)
        n_sigma: (vdim, )
    return :
        bounds: (N, vdim, 2)
        '''
    n_sigma = to_tensor(n_sigma, dtype=torch.float32, device=Sigma.device)
    offset = (n_sigma.unsqueeze(0) * Sigma) # (N, vdim)
    min_limits = torch.min(X0X1, dim=2).values - offset # torch.min(shape(N,vdim,2)) --> shape(N, vdim)
    max_limits = torch.max(X0X1, dim=2).values + offset
    return torch.stack([min_limits, max_limits], dim=2)


def _stack_X0X1(X0, X1):
    '''
    Arguments:
        X0: (N, vdim)
        X1: (N, vdim)
    Return:
       X0X1: (N, vdim, 2)
     '''
    return torch.stack((X0, X1), dim=2)


def compute_bounds_X0_X1(X0, X1, Sigma, n_sigma):
    '''
    Arguments:
        X0: (N, vdim)
        X1: (N, vdim)
        Sigma: (N, vdim)
        n_sigma: (vdim, )
    return:
        (N, vdim, 2) float
    '''
    n_sigma = to_tensor(n_sigma, dtype=torch.float32, device=Sigma.device)
    combined = _stack_X0X1(X0, X1)
    bounds = compute_bounds_X0X1(combined, Sigma, n_sigma)
    return bounds

def reduce_to_universal(shape: Tensor):
    """
    Compute a universal shape across all steps using reductions.

    Args:
        shapes (torch.Tensor): Shapes of the charge boxes (N, vdim).

    Returns:
        Universal shape. (vdim, )
    """
    universal_max = torch.max(shape, dim=0).values

    return universal_max


def compute_charge_box(X0: Tensor, X1: Tensor, Sigma: Tensor,
                       n_sigma, origin, grid_spacing, **kwargs):
    '''
    FIXME: Output is only meaningful for vdim == 3.

    Args:
        X0 : (N, vdim), Tensor
        X1 : (N, vdim), Tensor
        Sigma : (N, vdim)
        n_sigma : (vdim, )
        origin : (vdim, )
        grid_spacing : (vdim, )
    Returns:
        offset : (N, vdim)
        shape : (vdim,)
        '''
    recenter = kwargs.get('recenter', False)
    compare_key = kwargs.get('compare_key', 'index')

    device = X0.device

    n_sigma = to_tensor(n_sigma, device)

    extremes = compute_bounds_X0_X1(X0, X1, Sigma, n_sigma) # (N, vdim, 2)

    min_limit = extremes[:, :, 0] # (N, vdim)
    max_limit = extremes[:, :, 1] # (N, vdim)

    if compare_key == 'index':
        min_limit = compute_index(min_limit, origin, grid_spacing, device=device)
        max_limit = compute_index(max_limit, origin, grid_spacing, device=device)
        offset = min_limit

        shape = max_limit - min_limit + 1
        universal_shape = reduce_to_universal(shape)

        if recenter:
            # center moved to half the universal shape,
            # so the lower corner shifts by center - half_shape, which are negative values.
            # The new lower corner is smaller than old lower corner by |center - half_shape|
            # offset = offset + (center - half_shape)
            shift = (
                (shape - universal_shape)/2
            ).floor().to(index_dtype)
            offset = offset + shift
            assert torch.all(shift <= 0)

        universal_shape = universal_shape.to(index_dtype)

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
    return offset, universal_shape
