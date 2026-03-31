import torch
from torch import Tensor

from ..types import index_dtype, MAX_INDEX, MIN_INDEX
# from ..utils import to_tensor

import logging

logger = logging.getLogger('tred.raster.steps')

DEFAULT_FLOAT_DTYPE = torch.float64


def to_tensor(source, device, dtype=DEFAULT_FLOAT_DTYPE):
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


def compute_coordinate(idxs: Tensor, origin, grid_spacing, device='cpu',
                       dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Arguments:
        idxs : (N, vdim); index of grid point, Tensor of index_dtype
        origin : (vdim,), grid origin
        grid_spacing : (vdim, ), grid spacing
    return
        origin + spacing * idx
    '''
    fidxs = to_tensor(idxs, device=device, dtype=dtype)
    assert torch.all(fidxs <= MAX_INDEX), 'Overflow of index_dtype'
    assert torch.all(fidxs >= MIN_INDEX), 'Underflow of index_dtype'
    idxs = to_tensor(idxs, device=device, dtype=index_dtype)

    if idxs.dim() == 1:
        idxs = idxs.unsqueeze(0)
    origin = to_tensor(origin, device=idxs.device, dtype=dtype)
    grid_spacing = to_tensor(grid_spacing, device=device, dtype=dtype)
    return origin.unsqueeze(0) + idxs * grid_spacing.unsqueeze(0)


def compute_index(coords, origin, grid_spacing, device='cpu',
                  dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Arguments:
        coords : (N, vdim)
        origin : (vdim,)
        grid_spacing : (vdim, )

    return
        (coords - origin)//spacing, index of grid point
    '''
    coords = to_tensor(coords, device, dtype=dtype)
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
    origin = to_tensor(origin, device, dtype=dtype)
    grid_spacing = to_tensor(grid_spacing, device, dtype=dtype)
    idxs = (coords - origin.unsqueeze(0)) / grid_spacing.unsqueeze(0)

    assert torch.all(idxs <= MAX_INDEX), f'Overflow of index_dtype {MAX_INDEX}'
    assert torch.all(idxs >= MIN_INDEX), 'Underflow of index_dtype'
    return idxs.floor().to(index_dtype)


def compute_bounds_X0X1(X0X1, Sigma, n_sigma, dtype=DEFAULT_FLOAT_DTYPE):
    '''
        X0X1 : (N, vdim, 2)
        Sigma : (N, vdim)
        n_sigma: (vdim, )
    return :
        bounds: (N, vdim, 2)
        '''
    X0X1 = to_tensor(X0X1, device=Sigma.device, dtype=dtype)
    Sigma = to_tensor(Sigma, device=Sigma.device, dtype=dtype)
    n_sigma = to_tensor(n_sigma, dtype=dtype, device=Sigma.device)
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


def compute_bounds_X0_X1(X0, X1, Sigma, n_sigma, dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Arguments:
        X0: (N, vdim)
        X1: (N, vdim)
        Sigma: (N, vdim)
        n_sigma: (vdim, )
    return:
        (N, vdim, 2) float
    '''
    X0 = to_tensor(X0, device=Sigma.device, dtype=dtype)
    X1 = to_tensor(X1, device=Sigma.device, dtype=dtype)
    Sigma = to_tensor(Sigma, device=Sigma.device, dtype=dtype)
    n_sigma = to_tensor(n_sigma, dtype=dtype, device=Sigma.device)
    combined = _stack_X0X1(X0, X1)
    bounds = compute_bounds_X0X1(combined, Sigma, n_sigma, dtype=dtype)
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
                       n_sigma, origin, grid_spacing,
                       dtype=DEFAULT_FLOAT_DTYPE, **kwargs):
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
    X0 = to_tensor(X0, device=device, dtype=dtype)
    X1 = to_tensor(X1, device=device, dtype=dtype)
    Sigma = to_tensor(Sigma, device=device, dtype=dtype)

    n_sigma = to_tensor(n_sigma, device, dtype=dtype)

    extremes = compute_bounds_X0_X1(X0, X1, Sigma, n_sigma, dtype=dtype) # (N, vdim, 2)

    min_limit = extremes[:, :, 0] # (N, vdim)
    max_limit = extremes[:, :, 1] # (N, vdim)

    if compare_key == 'index':
        min_limit = compute_index(min_limit, origin, grid_spacing, device=device, dtype=dtype)
        max_limit = compute_index(max_limit, origin, grid_spacing, device=device, dtype=dtype)
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


def qpoint_diff3D(Q, X0, X1, Sigma, x, y, z):
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
    shape_new = [-1] + num_dims_to_add * [1]

    # Prepare for broadcasting
    x0, y0, z0 = tuple(X0[:, i].view(shape_new) for i in range(3))
    x1, y1, z1 = tuple(X1[:, i].view(shape_new) for i in range(3))
    sx, sy, sz = tuple(Sigma[:, i].view(shape_new) for i in range(3))
    Q = Q.view(shape_new)
    xc, yc, zc = (x0+x1)/2, (y0+y1)/2, (z0+z1)/2
    charge = Q / torch.pow(torch.tensor(2 * 3.1415926), 1.5) / sx / sy / sz \
        * torch.exp(- (x - xc)**2 / 2 / sx**2
                    - (y - yc)**2 / 2 / sy**2
                    - (z - zc)**2 / 2 / sz**2)
    return charge


def too_short(X0, X1, Sigma, threshold=0.05):
    #    sqrt2 = 1.4142135623730951
    x0, y0, z0 = [X0[:, i] for i in range(3)]
    x1, y1, z1 = [X1[:, i] for i in range(3)]
    sx, sy, sz = [Sigma[:, i] for i in range(3)]

    # Calculate differences
    dx01 = x0 - x1
    dy01 = y0 - y1
    dz01 = z0 - z1

    point_like = torch.stack(
        [torch.abs(dx01) < threshold * sx,
         torch.abs(dy01) < threshold * sy,
         torch.abs(dz01) < threshold * sz], dim=1
    )
    point_like = torch.all(point_like, dim=1)

    return point_like


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


def _create_w1d_GL(npt, spacing, device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        npt : integer
        spacing : float
    Return:
        a tensor of weights of n-point GL quadrature after correcting for length of intervals
    '''
    _, weights = roots_legendre(npt)
    w1d = torch.tensor(weights, dtype=dtype, requires_grad=False, device=device) * spacing/2
    return w1d

def _create_w1ds(method, npoints, grid_spacing, device='cpu',
                 dtype=DEFAULT_FLOAT_DTYPE):
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
        w1ds.append(_create_w1d_GL(npt, grid_spacing[ipt], device, dtype=dtype))
    return w1ds

def _create_w_block(w1ds):
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

def create_w_block(method, npoints, grid_spacing, device='cpu',
                   dtype=DEFAULT_FLOAT_DTYPE):
    '''create a weight block
    Args:
        method : str
        npoints : tuple[int], (vdim, ),
        grid_spacing : tuple[float], (vdim, )
    Returns:
        weights: Tensor, in a shape of (N_1, N_2, ..., N_i, ...) for i from 1 to vdim
    '''
    w1ds = _create_w1ds(method, npoints, grid_spacing, device, dtype=dtype)
    return _create_w_block(w1ds)


def _create_u1d_GL(npt, device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        npt : integer
    Returns:
        a tensor of coefficients for interpolations at roots of npt-order GL polynomials
    '''
    roots, _ = roots_legendre(npt)
    roots_tensor = torch.as_tensor(roots, dtype=dtype, device=device)
    zeros_tensor = torch.zeros_like(roots_tensor)

    negative_mask = roots_tensor < 0

    neg_u = torch.stack(
        [-roots_tensor / 2, 1 + roots_tensor / 2, zeros_tensor], dim=-1)

    pos_u = torch.stack([
        zeros_tensor, 1 - roots_tensor / 2, roots_tensor / 2], dim=-1)

    u = torch.where(negative_mask.unsqueeze(-1), neg_u, pos_u)
    u.requires_grad = False
    return u


def _create_u1ds(method, npoints, device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Arguments:
        method : str
        npoints : (vdim, ) integers
    Return:
        (Tensor_1, Tensor_2, Tensor_3), a list in which each elment is
          a tensor of coefficients for linear interpolations.
    '''
    u1ds = []
    for ipt, npt in enumerate(npoints):
        u1ds.append(_create_u1d_GL(npt, device, dtype=dtype))
    return u1ds

def _create_u_block(u1ds, interpo_npts=3):
    '''
    To create a weight block for u in 3D
    Requirements:
        u1d in u1ds is Tensor with a shape of (npt, interpo_npts) where npt means n-point GL quad rule.
    Args:
        u1ds: list, (Tensor_1, Tensor_2, ..., Tensor_i, ...)
    Return:
        A tesnor in a shape of (N_1, N_2, ..., N_i, ..., interpo_npts, interpo_npts, ..., interpo_npts_i, ...)
    '''
    ndim = len(u1ds)
    for i in range(ndim):
        if u1ds[i].shape[-1] != interpo_npts:
            raise ValueError(
                f'u1ds must have last dimension of size '
                f'{interpo_npts} for linear interpolation')
    # [-1, 1, ..., 3, 1, ...]
    shape_new = [-1,] + (ndim-1) * [1,] + [interpo_npts,] + (ndim-1) * [1]
    ublock = u1ds[0].view(shape_new)
    for i in range(1, ndim, 1):
        shape_new = [1, ] * ndim + [1, ] * ndim
        shape_new[i] = -1
        shape_new[ndim+i] = interpo_npts
        ublock = ublock * u1ds[i].view(shape_new)

    return ublock


def create_u_block(method, npoints, device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    To create a weight block for u in 3D
    Args:
        method: string
        npoints: tuple[int], (vdim, )
        device
    Return:
        A tesnor in a shape of (N_1, N_2, ..., N_i, ..., 2, 2, ..., 2_i, ...)

    '''
    u1ds = _create_u1ds(method, npoints, device, dtype=dtype)
    return _create_u_block(u1ds)


def _create_wu_block(w, u):
    '''
    Args:
        w3d (L,M,N,...)
        u3d (L,M,N,...,2,2,2,...)
    Return:
        return w3d[i,j,k,...] * u3d{i,j,k,...,:,:,:,...]
    '''
    ndim = w.ndim
    shape = tuple(w.size()) + ndim * (1,)
    if w.ndim * 2 != u.ndim:
        raise ValueError('w and u does not match on their dimensions;'
                         f' w size: {w.size()}, u size: {u.size()}')
    return w.view(shape) * u


def create_wu_block(method, npoints, grid_spacing, device='cpu',
                    dtype=DEFAULT_FLOAT_DTYPE):
    w = create_w_block(method, npoints, grid_spacing, device, dtype=dtype)
    u = create_u_block(method, npoints, device, dtype=dtype)
    return _create_wu_block(w, u)


def create_grid1d(origin_1d, grid_spacing_1d, offset_1d, ngrid_1d : int,
                  device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        origin: float or 0-dimensional Tensor
        grid_spacing: float or 0-dimensional Tensor
        offset: Tensor of index_type, (N, )
        ngrid_1d: integer, number of grid points
    Returns:
        grid points

    FIXME: shp may be overflow?
    '''
    # if isinstance(shp_1d, Tensor):
    #     shp_1d = shp_1d.item()
    # if shp_1d > MAX_INDEX:
    #     raise ValueError(f'shp_1d={shp_1d} is larger than MAX_INDEX {MAX_INDEX}.')

    ngrid_idx_1d = torch.arange(ngrid_1d, dtype=index_dtype, requires_grad=False,
                              device=device) # (shp_1d, )
    idx_2d = offset_1d[:,None] + ngrid_idx_1d[None, :]
    grid_1d = compute_coordinate(idx_2d, (origin_1d,), (grid_spacing_1d,),
                                 device=device, dtype=dtype)
    return grid_1d

def _create_node1d_GL(npt, origin_1d, grid_spacing_1d, offset_1d, shp_1d,
                      device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        origin_1d : float or 0-dimensional Tensor
        grid_spacing_1d : float or 0-dimensional Tensor
        offset_1d : (Nsteps, ) Tensor of integers
        shp_1d : integer, the length of grid along the axis, in term of grid spacing
        npt : integer, npoints GL quad rule
    return:
        nodes (Nsteps, npt, shp_1d-1).

    FIXME: shp may be overflow?
        '''
    # if isinstance(shp_1d, Tensor):
    #     shp_1d = shp_1d.item()
    # if shp_1d > MAX_INDEX:
    #     raise ValueError(f'shp_1d={shp_1d} is larger than MAX_INDEX {MAX_INDEX}.')

    grid_1d = create_grid1d(origin_1d, grid_spacing_1d, offset_1d, shp_1d+1,
                            device, dtype=dtype)
    roots, _ = roots_legendre(npt)
    roots = to_tensor(roots, device=device, dtype=dtype)
    half_delta = (grid_1d[:, 1:] - grid_1d[:, :-1])/2. # (Nsteps, shp_1d-1)
    avg = (grid_1d[:,1:] + grid_1d[:,:-1])/2. # (Nsteps, shp_1d-1)
    node_1d = half_delta[:, None, :] * roots[None, :, None] \
        + avg[:, None, :] # (Nsteps, npt, shp_1d-1)
    return node_1d


def create_node1ds(method, npoints, origin, grid_spacing, offset, shape,
                   device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        method : str
        origin : tuple[float], in a size of (vdim, )
        grid_spaing : tuple[float], in a size of (vdim, )
        offset : Tensor, in a size of (Nbatch, vdim)
        shape : tuple[int], in a size of (vdim, ),
                the length of grid along each axis, in term of grid spacing, (shape1, shape2, ...)
        npoints : (vdim, ), n points for GL quad rule, (npoints1, npoints2, ...)
        device : device
    Return:
        a list in which each element is a tensor with a shape
        of (Nsteps, npoints1, npoints2, ..., shape1, shape2, ...)
    '''
    node1ds = []
    if method != 'gauss_legendre':
        raise NotImplementedError('Not implemented method but gauss legendre quadrature')
    for i in range(len(origin)):
        node1ds.append(
            _create_node1d_GL(npoints[i], origin[i], grid_spacing[i],
                              offset[:,i], shape[i], device, dtype=dtype)
        )
    return node1ds


def eval_qmodel(Q, X0, X1, Sigma, x, y, z, qmodel=qline_diff3D, **kwargs):
    '''
    Args:
        Q (Nsteps, )
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        x, y, z are in a shape of (Nsteps, L/M/N, I/J/K-1)
    Return:
        q

    FIXME: to support axes with no nodes
    '''

    #FIXME: each axis must be associated with nodes of quadrature rules for now...
    charge = qmodel(Q, X0, X1, Sigma, x[:, :, None, None, :, None, None], # (Nsteps, L, 1, 1, I-1, 1, 1)
                    y[:, None, :, None, None, :, None], # (Nsteps, 1, M, 1, 1, J-1, 1)
                    z[:, None, None, :, None, None, :]) # (Nsteps, 1, 1, N, 1, 1, K-1)

    return charge


def eval_qeff(Q, X0, X1, Sigma, offset, shape, origin, grid_spacing,
              method, npoints, dtype=DEFAULT_FLOAT_DTYPE, **kwargs):
    '''
    Args:
        Q (Nsteps, )
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        offset (Nsteps, vdim)
        shape (vdim,)
        origin (vdim,)
        grid_spacing (vdim,)
        npoints (vdim, )
        kwargs:
            qmodel: method to calculate charges, taking (Q, X0, X1, Sigma, x, y, z)
                    as arguments. This argument will be passed to eval_qmodel
                    so that broadcasting is done properly.
    Return:
        effective charge

    FIXME:
       w block, u blocks supports vdim = any number,
       offset, shape, origin, they also support vdim = any number,
       Q, X0, X1, Sigma, their vdim is always 3.
    FIXME:
       vdim of npoints in principle can be different from offset, shape, origin, spacing.
       Now it is fixed to the same as others.
    '''
    if not isinstance(Q, torch.Tensor):
        raise ValueError('Q must be a torch.Tensor')
    device = Q.device
    Q = Q.to(device=device, dtype=dtype)
    X0 = X0.to(device=device, dtype=dtype)
    X1 = X1.to(device=device, dtype=dtype)
    Sigma = Sigma.to(device=device, dtype=dtype)
    offset = offset.to(device=device, dtype=index_dtype)
    shape = shape.to(device=device, dtype=index_dtype)
    grid_spacing = to_tensor(grid_spacing, device=device, dtype=dtype)
    origin = to_tensor(origin, device=device, dtype=dtype)

    qline_model = kwargs.get('qline_model', qline_diff3D)
    qpoint_model = kwargs.get('qpoint_model', qpoint_diff3D)
    threshold = kwargs.get('threshold', 0.1)

    # FIXME: dimensions are hard coded
    kernel = create_wu_block(method, npoints, grid_spacing, device, dtype=dtype)
    # it does not matter we flip at first or we multiply w and u at first
    kernel = torch.flip(kernel, [3, 4, 5])
    lmn = kernel.size()[:3]
    lmn_prod = lmn[0] * lmn[1] * lmn[2]
    rst = kernel.size()[3:]
    # out_channel, in_channel/groups, R, S, T
    kernel = kernel.view(lmn_prod, 1, rst[0], rst[1], rst[2])

    x, y, z = create_node1ds(method, npoints, origin, grid_spacing,
                             offset, shape, device, dtype=dtype)

    # FIXME: may update batch dimension in the future
    too_short_mask = too_short(X0, X1, Sigma, threshold=threshold)

    charge = torch.zeros((Q.size(0), *lmn, *shape),
                         dtype=dtype, device=device)
    mask_shape = [-1,] + [1,] * (len(lmn) + len(shape))
    expand_shape = [Q.size(0), *lmn] + [s.item() for s in shape]
    too_short_mask = too_short_mask.view(mask_shape).expand(*expand_shape)
    charge = torch.where(too_short_mask,
                         eval_qmodel(Q, X0, X1, Sigma, x, y, z,
                                     qmodel=qpoint_model, **kwargs),
                         eval_qmodel(Q, X0, X1, Sigma, x, y, z,
                                     qmodel=qline_model, **kwargs))

    charge = charge.view(Q.size(0), lmn_prod,  # batch, chanenel
                         shape[0], shape[1], shape[2])  # D1, D2, D3
    # FIXME: the padding is only valid for interpolation points == 3
    charge = torch.nn.functional.conv3d(charge, kernel, padding='same',
                                        groups=lmn_prod)
    charge = charge.sum(dim=1)
    return charge, offset


def compute_qeff(Q, X0, X1, Sigma, n_sigma, origin, grid_spacing, method, npoints, **kwargs):
    '''
    See compute_charge_box, eval_qeff,
    Return:
        qeff, offset
    '''
    if kwargs.get('recenter', False) or kwargs.get('skippad', False):
        if not kwargs.get('recenter', False):
            kwargs['recenter'] = True
            logger.warning('Option "recenter" is overwritten to True, complying with option "skippad"')
        if not kwargs.get('skippad', False):
            kwargs['skippad'] = True
            logger.warning('Option "skippad" is overwritten to True, complying with option "recenter"')
    dtype = kwargs.pop('dtype', DEFAULT_FLOAT_DTYPE)
    Q = Q.to(dtype=dtype)
    X0 = X0.to(dtype=dtype)
    X1 = X1.to(dtype=dtype)
    Sigma = Sigma.to(dtype=dtype)
    # X0, X1 shifted by half bin, as in responses.org
    # FIXME: how to handle the case when X0,X1 hit the anode?
    offset, shape = compute_charge_box(X0, X1, Sigma, n_sigma, origin, grid_spacing,
                                       dtype=dtype, **kwargs)
    qeff, offset = eval_qeff(Q, X0, X1, Sigma, offset, shape, origin, grid_spacing,
                             method, npoints, dtype=dtype, **kwargs)
    return qeff, offset
