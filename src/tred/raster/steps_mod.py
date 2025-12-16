import tred.raster.steps as steps_mod
from tred.raster.steps import (create_wu_block, create_node1ds, create_w_block,
                               eval_qmodel, qline_diff3D, roots_legendre)
from tred.raster.steps import _create_u_block as steps__create_u_block
from tred.raster.steps import _create_u1d_GL as steps__create_u1d_GL
from tred.raster.steps import eval_qeff as steps__eval_qeff
import torch


def _create_u1d_GL(npt, device='cpu'):
    roots, _ = roots_legendre(npt)
    roots = torch.as_tensor(roots).to(torch.float64).to(device)
    u = torch.where((roots < 0).unsqueeze(-1),
                    torch.stack([-roots/2, 1+roots / 2,
                                 torch.zeros(npt, device=device,
                                             dtype=torch.float64)], dim=-1),
                    torch.stack([torch.zeros(npt, device=device,
                                             dtype=torch.float64),
                                 1 - roots / 2, roots / 2], dim=-1))
    return u


def _create_u_block(u1ds):
    '''
    To create a weight block for u in 3D
    Requirements:
        u1d in u1ds is Tensor with a shape of (npt, 2) where npt means n-point GL quad rule.
    Args:
        u1ds: list, (Tensor_1, Tensor_2, ..., Tensor_i, ...)
    Return:
        A tesnor in a shape of (N_1, N_2, ..., N_i, ..., 3, 3, ..., 3_i, ...)
    '''
    ndim = len(u1ds)
    interpo_npts = 3
    # [-1, 1, ..., 2, 1, ...]
    shape_new = [-1,] + (ndim-1) * [1,] + [interpo_npts,] + (ndim-1) * [1]
    ublock = u1ds[0].view(shape_new)
    for i in range(1, ndim, 1):
        shape_new = [1, ] * ndim + [1, ] * ndim
        shape_new[i] = -1
        shape_new[ndim+i] = interpo_npts
        ublock = ublock * u1ds[i].view(shape_new)

    return ublock


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


def too_small(X0, X1, Sigma, threshold=0.05):
    #    sqrt2 = 1.4142135623730951
    x0, y0, z0 = [X0[:, i] for i in range(3)]
    x1, y1, z1 = [X1[:, i] for i in range(3)]
    sx, sy, sz = [Sigma[:, i] for i in range(3)]

    # Calculate differences
    dx01 = x0 - x1
    dy01 = y0 - y1
    dz01 = z0 - z1

    # Calculate squared terms
    sxsy2 = (sx*sy)**2
    sxsz2 = (sx*sz)**2
    sysz2 = (sy*sz)**2

    # Calculate delta terms
    deltaSquare = (
        sysz2 * dx01**2 +
        sxsy2 * dz01**2 +
        sxsz2 * dy01**2
    )
    deltaSquareSqrt = torch.sqrt(deltaSquare)

    return torch.all(deltaSquareSqrt[:, None] < Sigma * threshold, dim=1)


def eval_qeff(Q, X0, X1, Sigma, offset, shape, origin, grid_spacing,
              method, npoints, **kwargs):
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

    qmodel = kwargs.get('qmodel', qline_diff3D)
    threshold = kwargs.get('threshold', 0.05)

    # FIXME: not friendly to JIT

    # FIXME: dimensions are hard coded
    kernel = create_wu_block(method, npoints, grid_spacing, device)
    # it does not matter we flip at first or we multiply w and u at first
    kernel = torch.flip(kernel, [3, 4, 5])
    lmn = kernel.size()[:3]
    lmn_prod = lmn[0] * lmn[1] * lmn[2]
    rst = kernel.size()[3:]
    # out_channel, in_channel/groups, R, S, T
    kernel = kernel.view(lmn_prod, 1, rst[0], rst[1], rst[2])

    x, y, z = create_node1ds(method, npoints, origin, grid_spacing,
                             offset, shape, device)

    # FIXME: may update batch dimension in the future
    too_small_mask = too_small(X0, X1, Sigma, threshold=threshold)

    charge = torch.zeros((Q.size(0), *lmn, *shape),
                         dtype=torch.float64, device=device)
    mask_shape = [-1,] + [1,] * (len(lmn) + len(shape))
    expand_shape = [Q.size(0), *lmn] + [s.item() for s in shape]
    too_small_mask = too_small_mask.view(mask_shape).expand(*expand_shape)
    charge = torch.where(too_small_mask,
                         eval_qmodel(Q, X0, X1, Sigma, x, y, z,
                                     qmodel=qpoint_diff3D, **kwargs),
                         eval_qmodel(Q, X0, X1, Sigma, x, y, z,
                                     qmodel=qmodel, **kwargs))

    charge = charge.view(Q.size(0), lmn_prod,  # batch, chanenel
                         shape[0], shape[1], shape[2])  # D1, D2, D3

    charge = torch.nn.functional.conv3d(charge, kernel, padding='same',
                                        groups=lmn_prod)
    charge = charge.sum(dim=1)
    return charge, offset


def eval_q(Q, X0, X1, Sigma, offset, shape, origin, grid_spacing,
           method, npoints, **kwargs):
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

    qmodel = kwargs.get('qmodel', qline_diff3D)
    threshold = kwargs.get('threshold', 0.05)

    # FIXME: not friendly to JIT

    # FIXME: dimensions are hard coded
    kernel = create_w_block(method, npoints, grid_spacing, device)
    # it does not matter we flip at first or we multiply w and u at first
    lmn = kernel.size()
    lmn_prod = lmn[0] * lmn[1] * lmn[2]

    x, y, z = create_node1ds(method, npoints, origin, grid_spacing,
                             offset, shape, device)

    # FIXME: may update batch dimension in the future
    too_small_mask = too_small(X0, X1, Sigma, threshold=threshold)

    charge = torch.zeros((Q.size(0), *lmn, *shape),
                         dtype=torch.float64, device=device)
    mask_shape = [-1,] + [1,] * (len(lmn) + len(shape))
    expand_shape = [Q.size(0), *lmn] + [s.item() for s in shape]
    too_small_mask = too_small_mask.view(mask_shape).expand(*expand_shape)

    charge = torch.where(too_small_mask,
                         eval_qmodel(Q, X0, X1, Sigma, x, y, z,
                                     qmodel=qpoint_diff3D, **kwargs),
                         eval_qmodel(Q, X0, X1, Sigma, x, y, z,
                                     qmodel=qmodel, **kwargs))
    charge[:] = charge * kernel.view(1, *lmn, 1, 1, 1)
    charge = charge.view(Q.size(0), lmn_prod,  # batch, chanenel
                         shape[0], shape[1], shape[2])  # D1, D2, D3
    charge = charge.sum(dim=1)

    return charge, offset

def add_patches():
    steps_mod._create_u1d_GL = _create_u1d_GL
    steps_mod._create_u_block = _create_u_block
    steps_mod._eval_qeff = eval_qeff

def remove_patches():
    steps_mod._create_u1d_GL = steps__create_u1d_GL
    steps_mod._create_u_block = steps__create_u_block
    steps_mod._eval_qeff = steps__eval_qeff
