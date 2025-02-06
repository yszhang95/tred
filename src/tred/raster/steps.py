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


def qline_diff3D(Q, X0, X1, Sigma, x, y, z):
    """
    Args:
        Q (N,)
        X0 (N, 3)
        X1 (N, 3)
        Sigma (N, 3)
        x (N, other shape)
        y (N, other shape)
        z (N, other shape)
    return:
        q (N, othter shape)
    """
    num_dims_to_add = x.ndim - 1
    shape_new = [-1] + num_dims_to_add * [1] # TorchScript does NOT support (-1,) + dynamic tuples

    # Prepare for broadcasting
    x0, y0, z0 = tuple(X0[:,i].view(shape_new) for i in range(3))
    x1, y1, z1 = tuple(X1[:,i].view(shape_new) for i in range(3))
    sx, sy, sz = tuple(Sigma[:,i].view(shape_new) for i in range(3))
    Q = Q.view(shape_new)
    args = (Q, x0, y0, z0, x1, y1, z1, sx, sy, sz, x, y, z)
    return qline_diff3D_script(*args)


@torch.jit.script
def qline_diff3D_script(
        Q: Tensor, x0: Tensor, y0: Tensor, z0: Tensor,
        x1: Tensor, y1: Tensor, z1: Tensor,
        sx: Tensor, sy: Tensor, sz: Tensor,
        x, y, z
) -> Tensor:
    '''
    Args:
        Q: (N, 1, 1, 1, ...)
        x0, y0, z0, x1, y1, z1, sx, sy, sz: (N, 1, 1, 1, ...)
        x (N, other shape)
        y (N, other shape)
        z (N, other shape)
    return:
        q (N, othter shape)
    '''
    sqrt2 = 1.4142135623730951

    # Calculate differences
    dx01 = x0 - x1
    dy01 = y0 - y1
    dz01 = z0 - z1

    # Calculate squared terms
    sxsy2 = (sx*sy)**2
    sxsz2 = (sx*sz)**2
    sysz2 = (sy*sz)**2
    sx2 = sx**2
    sy2 = sy**2
    sz2 = sz**2

    # Calculate delta terms
    deltaSquare = (
        sysz2 * dx01**2 +
        sxsy2 * dz01**2 +
        sxsz2 * dy01**2
    )
    deltaSquareSqrt = torch.sqrt(deltaSquare)

    # Calculate charge distribution
    QoverDeltaSquareSqrt4pi = Q / (deltaSquareSqrt * 4 * torch.pi)
    erfArgDenominator = sqrt2 * deltaSquareSqrt * sx * sy * sz

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


try:
    import scipy
    roots_legendre = scipy.special.roots_legendre
except:
    def roots_legendre(n: int) -> tuple[tuple[float]]:
        '''
        A lookup table for GL rule.
        generated by the following
        import scipy
        for i in range(1,7):
            roots, weights = scipy.special.roots_legendre(i)
            roots_str = '(' + ', '.join(f'{x:.16f}' for x in roots) + ')'
            weights_str = '(' + ', '.join(f'{x:.16f}' for x in weights) + ')'
            print(f'if n == {i}:\n    return {roots_str}, {weights_str}')
        '''
        if n == 1:
            return (0.0000000000000000,), (2.0000000000000000,)
        if n == 2:
            return (-0.5773502691896258, 0.5773502691896258), \
                (1.0000000000000000, 1.0000000000000000)
        if n == 3:
            return (-0.7745966692414834, 0.0000000000000000, 0.7745966692414834), \
                (0.5555555555555558, 0.8888888888888883, 0.5555555555555558)
        if n == 4:
            return (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526), \
                (0.3478548451374536, 0.6521451548625464, 0.6521451548625464, 0.3478548451374536)
        if n == 5:
            return (-0.9061798459386640, -0.5384693101056831, 0.0000000000000000, 0.5384693101056831, 0.9061798459386640), \
                (0.2369268850561896, 0.4786286704993662, 0.5688888888888886, 0.4786286704993662, 0.2369268850561896)
        if n == 6:
            return (-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.9324695142031521), \
                (0.1713244923791695, 0.3607615730481392, 0.4679139345726914, 0.4679139345726914, 0.3607615730481392, 0.1713244923791695)
        raise ValueError(f"customized roots_legendre does not support n = {n}"\
                     " (must be 1 <= n <= 6)")

def _create_w1d_GL(npt, spacing, device='cpu'):
    '''
    Args:
        npt : integer
        spacing : float
    Return:
        a tensor of weights of n-point GL quadrature after correcting for length of intervals
    '''
    _, weights = roots_legendre(npt)
    w1d = torch.tensor(weights, dtype=torch.float32, requires_grad=False, device=device) * spacing/2
    return w1d

def _create_w1ds(method, npoints, grid_spacing, device='cpu'):
    '''
    Args:
        method : str
        npoints : (vdim, ), integers
        grid_spacing : (vdim, ), float
    return
        (vdim, ), a list in which each element is a tensor of weights of
        n-point GL quadrature after correcting for length of intervals
    '''
    if method != 'gauss_legendre':
        raise NotImplementedError('Not implemented method but gauss legendre quadrature')
    w1ds = []
    for ipt, npt in enumerate(npoints):
        w1ds.append(_create_w1d_GL(npt, grid_spacing[ipt], device))
    return w1ds

def _create_weight_block(w1ds):
    '''create a weight block
    Args:
        w1ds: tuple[Tensor, Tensor, ...], (Tensor_1, Tensor_2, ...), length of tuple is vdim,
    Return:
        weights: Tensor, in a shape of (N_1, N_2, ..., N_i, ...) for i from 1 to vdim
    '''
    ndim = len(w1ds)
    shape_0 = [-1,] + (ndim-1) * [1,] # special case [-1] + 0 * [1] --> [-1]
    wblock = w1ds[0].view(shape_0)
    for i in range(1, ndim, 1):
        shape_new = [1,] * ndim
        shape_new[i] = -1
        wblock = wblock * w1ds[i].view(shape_new)
    return wblock

def create_weight_block(method, npoints, grid_spacing, device='cpu'):
    '''create a weight block
    Args:
        method : str
        npoints : (vdim, ), integers
        grid_spacing : (vdim, ), float
    Returns:
        weights: Tensor, in a shape of (N_1, N_2, ..., N_i, ...) for i from 1 to vdim
    '''
    w1ds = _create_w1ds(method, npoints, grid_spacing, device)
    return _create_weight_block(w1ds)
