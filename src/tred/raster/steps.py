import torch

from ..types import index_dtype, MAX_INDEX, MIN_INDEX
# from ..utils import to_tensor


def to_tensor(source, device, dtype=torch.float32):
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


def compute_coordinate(idxs, origin, grid_spacing, device='cpu'):
    '''
    Arguments:
        idxs : (N, vdim); index of grid point
        origin : (vdim,), grid origin
        grid_spacing : (vdim, ), grid spacing
    return
        origin + spacing * idx
    '''
    fidxs = to_tensor(idxs, device=device, dtype=torch.float32)
    assert torch.any(fidxs <= MAX_INDEX), 'Overflow of index_type'
    assert torch.any(fidxs >= MIN_INDEX), 'Underflow of index_type'
    idxs = to_tensor(idxs, device=device, dtype=index_dtype)

    if idxs.dim() == 1:
        idxs = idxs.unsqueeze(0)
    origin = to_tensor(origin, device=device, dtype=torch.float32)
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

    assert torch.any(idxs <= MAX_INDEX), 'Overflow of index_type'
    assert torch.any(idxs >= MIN_INDEX), 'Underflow of index_type'

    return idxs.floor().to(index_dtype)
