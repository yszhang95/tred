import tred.raster.steps as ts
from tred.raster.steps import (
    compute_index, compute_coordinate, compute_charge_box,
    qline_diff3D, qpoint_diff3D, roots_legendre,
)

import json
import logging
import math
import sys
from pathlib import Path

import torch

logger = logging.getLogger('tred/tests/effq/test_effq.py')

THREE_SIGMA = (3, 3, 3)
CHARGE_RECOVERY_REL_TOL = 5e-3
QPOINT_CHARGE_RECOVERY_REL_TOL = 7e-3
TEST_Q_VALUES = (0.25, 1.0, 2.5)


def assert_charge_recovery(effq, charge, rel_tol=CHARGE_RECOVERY_REL_TOL):
    charge = charge.to(device=effq.device, dtype=effq.dtype)
    recovered = torch.sum(effq, dim=tuple(range(1, effq.ndim)))
    charge_sum = charge.reshape(charge.shape[0], -1).sum(dim=1)
    denom = torch.clamp(torch.abs(charge_sum), min=torch.finfo(effq.dtype).eps)
    rel_err = torch.abs(recovered - charge_sum) / denom
    max_err, max_idx = torch.max(rel_err, dim=0)
    msg = (
        f'max relative error {max_err.item()} at batch index {max_idx.item()}: '
        f'recovered {recovered[max_idx].item()} vs input {charge_sum[max_idx].item()}'
    )
    assert torch.all(rel_err < rel_tol), msg

def test_QModel():
    local_logger = logger.getChild('test_QModel')
    local_logger.debug('Testing qmodel by comparing results with Mathematica')
    Q = torch.tensor([1]).view(1)
    X0 = torch.tensor((0.4,2.4,3.4), dtype=torch.float64).view(1, 3)
    X1 = torch.tensor((0.6, 2.6, 3.6), dtype=torch.float64).view(1, 3)
    Sigma = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64).view(1,3)

    x = torch.linspace(0.2, 0.8, 5, dtype=torch.float64).view(1,-1)
    y = torch.linspace(2.2, 2.8, 5, dtype=torch.float64).view(1,-1)
    z = torch.linspace(3.2, 3.8, 5, dtype=torch.float64).view(1,-1)

    testq = qline_diff3D(Q, X0, X1, Sigma,
                                x.unsqueeze(2).unsqueeze(2),
                                y.unsqueeze(2).unsqueeze(1),
                                z.unsqueeze(1).unsqueeze(1))
    with Path(__file__).with_name('exact_qline_gaus.json').open() as f:
        exact = json.load(f)
    for i in range(testq.shape[1]):
        for j in range(testq.shape[2]):
            for k in range(testq.shape[3]):
                d = abs(testq[0,i,j,k].item()-exact[i][j][k])
                msg = f'difference = {d} at index = ({i},{j},{k}), '\
                    f'(x,y,z)={x[0,i].item(),y[0,j].item(),z[0,k].item()}), '\
                    f'testq = {testq[0,i,j,k].item()}, q from mathematica = {exact[i][j][k]}'
                assert torch.isclose(testq[0,i,j,k], torch.tensor(exact[i][j][k], dtype=torch.float64),
                                     atol=1E-12, rtol=1E-12), msg


def test_qpoint_diff3D_matches_erf_box_integral():
    Q = torch.tensor([1.0], dtype=torch.float64)
    center = torch.tensor([[0.5, 2.5, 3.5]], dtype=torch.float64)
    Sigma = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    origin = (0, 0, 0)
    grid_spacing = (0.1, 0.1, 0.1)
    offset = torch.tensor([(-10, 10, 20)], dtype=torch.int32)
    shape = (30, 30, 30)
    npt = (4, 4, 4)

    x, y, z = ts.create_node1ds(
        'gauss_legendre', npt, origin, grid_spacing, offset, shape, dtype=torch.float64
    )
    q = ts.eval_qmodel(Q, center, center, Sigma, x, y, z, qmodel=qpoint_diff3D)
    w = ts.create_w_block('gauss_legendre', npt, grid_spacing, dtype=torch.float64)
    approx = torch.sum(w[None, ..., None, None, None] * q)

    lower = ts.compute_coordinate(offset, origin, grid_spacing, dtype=torch.float64)[0]
    upper = ts.compute_coordinate(
        offset + torch.tensor(shape, dtype=torch.int32), origin, grid_spacing, dtype=torch.float64
    )[0]
    exact = Q[0].item()
    for lo, hi, mu, sigma in zip(
        lower.tolist(), upper.tolist(), center[0].tolist(), Sigma[0].tolist()
    ):
        exact *= 0.5 * (
            math.erf((hi - mu) / (math.sqrt(2) * sigma))
            - math.erf((lo - mu) / (math.sqrt(2) * sigma))
        )

    assert torch.isfinite(approx)
    assert torch.isclose(approx, torch.tensor(exact, dtype=approx.dtype), atol=1E-10, rtol=1E-7)


def test_eval_qeff_qpoint_charge_recovery():
    npt = (4, 4, 4)
    method = 'gauss_legendre'
    origin = (0, 0, 0)
    grid_spacing = (0.1, 0.1, 0.1)
    Q = torch.tensor(TEST_Q_VALUES, dtype=torch.float64)
    X0 = torch.tensor([(0.5, 2.5, 3.5)] * len(Q), dtype=torch.float64)
    X1 = X0.clone()
    Sigma = torch.tensor([(0.5, 0.5, 0.5)] * len(Q), dtype=torch.float64)

    offset, shape = ts.compute_charge_box(
        X0, X1, Sigma, THREE_SIGMA, origin, grid_spacing, dtype=torch.float64
    )
    qeff_default, _ = ts.eval_qeff(
        Q=Q, X0=X0, X1=X1, Sigma=Sigma, offset=offset, shape=shape,
        origin=origin, grid_spacing=grid_spacing, method=method, npoints=npt,
        dtype=torch.float64,
    )
    qeff_point, _ = ts.eval_qeff(
        Q=Q, X0=X0, X1=X1, Sigma=Sigma, offset=offset, shape=shape,
        origin=origin, grid_spacing=grid_spacing, method=method, npoints=npt,
        dtype=torch.float64, threshold=10.0,
    )

    assert_charge_recovery(qeff_default, Q, rel_tol=QPOINT_CHARGE_RECOVERY_REL_TOL)
    assert_charge_recovery(qeff_point, Q, rel_tol=QPOINT_CHARGE_RECOVERY_REL_TOL)
    assert torch.allclose(qeff_default, qeff_point, atol=1E-12, rtol=1E-12)


def test_qline_and_qpoint_agree_for_short_segment():
    npt = (4, 4, 4)
    method = 'gauss_legendre'
    origin = (0, 0, 0)
    grid_spacing = (0.1, 0.1, 0.1)
    Q = torch.tensor([1.0], dtype=torch.float64)
    X0 = torch.tensor([[0.495, 2.5, 3.5]], dtype=torch.float64)
    X1 = torch.tensor([[0.505, 2.5, 3.5]], dtype=torch.float64)
    Sigma = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)

    offset, shape = ts.compute_charge_box(
        X0, X1, Sigma, THREE_SIGMA, origin, grid_spacing, dtype=torch.float64
    )
    qeff_line, _ = ts.eval_qeff(
        Q=Q, X0=X0, X1=X1, Sigma=Sigma, offset=offset, shape=shape,
        origin=origin, grid_spacing=grid_spacing, method=method, npoints=npt,
        dtype=torch.float64, threshold=0.0,
    )
    qeff_point, _ = ts.eval_qeff(
        Q=Q, X0=X0, X1=X1, Sigma=Sigma, offset=offset, shape=shape,
        origin=origin, grid_spacing=grid_spacing, method=method, npoints=npt,
        dtype=torch.float64, threshold=10.0,
    )

    assert_charge_recovery(qeff_point, Q, rel_tol=QPOINT_CHARGE_RECOVERY_REL_TOL)
    assert torch.allclose(qeff_line, qeff_point, atol=1E-6, rtol=1E-5)

def test_roots_legendre():
    local_logger = logger.getChild('test_roots_legendre')
    local_logger.debug('Testing roots_legendre')
    for i in range(1, 5):
        roots, weights = roots_legendre(i)
        rstr = 'roots = (' + ', '.join(f'{x:.16f}' for x in roots) + ')'
        wstr = 'weights = (' + ', '.join(f'{x:.16f}' for x in weights) + ')'
        local_logger.debug(f'{i}-th order, {rstr}, {wstr}')
    try:
        roots, weights = roots_legendre(10)
    except ValueError as e:
        local_logger.debug('Does not support order 10')
        local_logger.debug(repr(e))


def test_create_w1d_GL(level=None):
    local_logger = logger.getChild('test_create_w1d_GL')
    if level:
        local_logger.setLevel(level)
    try:
        import numpy as np
        import scipy as sp
        assert np.allclose(np.array(ts._create_w1d_GL(4, 0.1)), sp.special.roots_legendre(4)[1] * 0.05,
                           atol=1E-10, rtol=1E-5)
        local_logger.debug('pass assertion at rel. delta < 1E-5, given 4-point GL, spacing = 0.1')
    except ImportError as e:
        local_logger.warning('Numpy or scipy is not available. Bypass test_create_w1d_GL')
        local_logger.warning(repr(e))

def test_create_w1ds(level=None):
    local_logger = logger.getChild('test_create_w1ds')
    if level:
        local_logger.setLevel(level)
    test_create_w1d_GL(logging.CRITICAL)
    spacing = (0.1, 0.1, 0.1)
    npoints = (4, 4, 4)
    out3d = ts._create_w1ds('gauss_legendre', npoints, spacing)
    local_logger.debug(
        f'Testing {npoints} weights calculation in 1D for {len(npoints)} dims,'\
        f' interval width == {spacing}'
    )

    for i in range(len(out3d)):
        assert torch.allclose(ts._create_w1d_GL(npoints[i], spacing[i]),
                              out3d[i], atol=1E-10, rtol=1E-5)

    spacing = (0.1, 0.1,)
    npoints = (4, 4,)
    out3d = ts._create_w1ds('gauss_legendre', npoints, spacing)
    local_logger.debug(
        f'Testing {npoints} weights calculation in 1D for {len(npoints)} dims,'\
        f' interval width == {spacing}'
    )

    for i in range(len(out3d)):
        assert torch.allclose(ts._create_w1d_GL(npoints[i], spacing[i]),
                              out3d[i], atol=1E-10, rtol=1E-5
                              )
    local_logger.debug('pass assertion with rel. delta <1E-5')

def test_create_w_block(level=None):
    test_create_w1ds(logging.CRITICAL)

    local_logger = logger.getChild('test_create_w_block')
    if level:
        local_logger.setLevel(level)
    npoints = (3, 2, 1)
    spacing = (2., 2., 2.)
    local_logger.debug(
        f'Testing {npoints}-point weight block calculation in {len(npoints)}D,'
        f' interval width == {spacing}'
    )
    wblock  = ts.create_w_block('gauss_legendre', npoints, spacing)
    w1ds = ts._create_w1ds('gauss_legendre', npoints, spacing)
    for i in range(npoints[0]):
        for j in range(npoints[1]):
            for k in range(npoints[2]):
                assert torch.allclose(w1ds[0][i] * w1ds[1][j] * w1ds[2][k],
                                      wblock[i,j,k], atol=1E-10, rtol=1E-5)
    npoints = (3, 2,)
    spacing = (2., 2.,)
    local_logger.debug(
        f'Testing {npoints}-point weight block calculation in {len(npoints)}D,'
        f' interval width == {spacing}'
    )
    wblock  = ts.create_w_block('gauss_legendre', npoints, spacing)
    w1ds = ts._create_w1ds('gauss_legendre', npoints, spacing)
    for i in range(npoints[0]):
        for j in range(npoints[1]):
            assert torch.allclose(w1ds[0][i] * w1ds[1][j],
                                  wblock[i,j], atol=1E-10, rtol=1E-5)

    npoints = (3, )
    spacing = (2.,)
    local_logger.debug(
        f'Testing {npoints}-point weight block calculation in {len(npoints)}D,'
        f' interval width == {spacing}'
    )
    wblock  = ts.create_w_block('gauss_legendre', npoints, spacing)
    w1ds = ts._create_w1ds('gauss_legendre', npoints, spacing)
    for i in range(npoints[0]):
        assert torch.allclose(w1ds[0][i],
                              wblock[i], atol=1E-10, rtol=1E-5)

    local_logger.debug('pass rel assertion with rel. delta <1E-5')


def test_create_u1d_GL(level=None):
    local_logger = logger.getChild('test_create_u1d_GL')
    if level:
        local_logger.setLevel(level)

    npt = 4
    local_logger.debug(
        'Testing coefficient calculations of linear interpolations '
        f'on {npt}-point GL roots.'
    )

    import scipy as sp

    u1d = ts._create_u1d_GL(npt)
    roots = torch.as_tensor(sp.special.roots_legendre(npt)[0], dtype=u1d.dtype)
    expected = torch.where(
        (roots < 0).unsqueeze(-1),
        torch.stack(
            (-roots / 2, 1 + roots / 2, torch.zeros_like(roots)),
            dim=-1,
        ),
        torch.stack(
            (torch.zeros_like(roots), 1 - roots / 2, roots / 2),
            dim=-1,
        ),
    )
    assert torch.allclose(u1d, expected, atol=1E-10, rtol=1E-5)
    assert torch.allclose(
        torch.sum(u1d, dim=-1),
        torch.ones(npt, dtype=u1d.dtype),
        atol=1E-10,
        rtol=1E-5,
    )
    local_logger.debug(f'pass assertion at rel. delta < 1E-5, given {npt}-point GL')

def test_create_u1ds(level=None):
    test_create_u1d_GL(logging.CRITICAL)

    local_logger = logger.getChild('test_create_u1ds')
    if level:
        local_logger.setLevel(level)
    npoints = (4, 4, 4)
    out3d = ts._create_u1ds('gauss_legendre', npoints)
    local_logger.debug(
        f'Testing coefficient calculation of linear interpolations in 1D {npoints}-point GL roots '
        f'for {len(npoints)} dims'
    )

    for i in range(len(out3d)):
        assert torch.allclose(ts._create_u1d_GL(npoints[i]),
                              out3d[i], atol=1E-10, rtol=1E-5)

    npoints = (4, 4,)
    out3d = ts._create_u1ds('gauss_legendre', npoints)
    local_logger.debug(
        f'Testing coefficient calculation of linear interpolations in 1D {npoints}-point GL roots '
        f'for {len(npoints)} dims'
    )

    for i in range(len(out3d)):
        assert torch.allclose(ts._create_u1d_GL(npoints[i]),
                              out3d[i], atol=1E-10, rtol=1E-5)

    local_logger.debug('pass assertion with rel. delta <1E-5')

def test_create_u_block(level=None):
    test_create_u1ds(logging.CRITICAL)

    local_logger = logger.getChild('test_create_u_block')
    if level:
        local_logger.setLevel(level)

    npoints = (3, 2, 1)
    local_logger.debug(
        f'Testing u block calculation in {len(npoints)}D on {npoints}-point GL'
    )
    ublock  = ts.create_u_block('gauss_legendre', npoints)
    u1ds = ts._create_u1ds('gauss_legendre', npoints)
    for i in range(npoints[0]):
        for j in range(npoints[1]):
            for k in range(npoints[2]):
                assert torch.allclose(u1ds[0][i][:,None,None] * u1ds[1][j][None,:,None]
                                      * u1ds[2][k][None,None,:],
                                      ublock[i,j,k], atol=1E-10, rtol=1E-5)
    npoints = (3, 2,)
    spacing = (2., 2.,)
    local_logger.debug(
        f'Testing u block calculation in {len(npoints)}D on {npoints}-point GL'
    )
    ublock  = ts.create_u_block('gauss_legendre', npoints)
    u1ds = ts._create_u1ds('gauss_legendre', npoints)
    for i in range(npoints[0]):
        for j in range(npoints[1]):
            assert torch.allclose(u1ds[0][i][:,None] * u1ds[1][j][None,:],
                                  ublock[i,j], atol=1E-10, rtol=1E-5)

    npoints = (3, )
    spacing = (2.,)
    local_logger.debug(
        f'Testing u block calculation in {len(npoints)}D on {npoints}-point GL'
    )

    ublock  = ts.create_u_block('gauss_legendre', npoints)
    u1ds = ts._create_u1ds('gauss_legendre', npoints)
    for i in range(npoints[0]):
        assert torch.allclose(u1ds[0][i],
                              ublock[i], atol=1E-10, rtol=1E-5)

    local_logger.debug('pass rel assertion with rel. delta <1E-5')


def test_create_wu_block(level=None):
    test_create_w_block(logging.CRITICAL)
    test_create_u_block(logging.CRITICAL)

    local_logger = logger.getChild('test_create_wu_block')
    if level:
        local_logger.setLevel(level)

    method = 'gauss_legendre'
    npoints = (1, 2, 1)
    spacing = (1., 1., 1.)
    local_logger.debug(f'Testing {npoints}-point wu block calculation in {len(npoints)}D,'
                       f' interval width == {spacing}')

    w = ts.create_w_block(method, npoints, spacing)
    u = ts.create_u_block(method, npoints)
    wu = ts.create_wu_block(method, npoints, spacing)
    for l in range(npoints[0]):
        for m in range(npoints[1]):
            for n in range(npoints[2]):
                for r in range(2):
                    for s in range(2):
                        for t in range(2):
                            assert torch.allclose(wu[l,m,n,r,s,t],
                                                  w[l,m,n]*u[l,m,n,r,s,t], atol=1E-10, rtol=1E-5)
    npoints = (2, 2)
    spacing = (1., 0.5)
    local_logger.debug(f'Testing {npoints}-point wu block calculation in {len(npoints)}D,'
                       f' interval width == {spacing}')

    w = ts.create_w_block(method, npoints, spacing)
    u = ts.create_u_block(method, npoints)
    wu = ts.create_wu_block(method, npoints, spacing)
    for l in range(npoints[0]):
        for m in range(npoints[1]):
            for r in range(2):
                for s in range(2):
                    assert torch.allclose(wu[l,m,r,s],
                                          w[l,m]*u[l,m,r,s], atol=1E-10, rtol=1E-5)
    npoints = (3,)
    spacing = (1.,)
    local_logger.debug(f'Testing {npoints}-point wu block calculation in {len(npoints)}D,'
                       f' interval width == {spacing}')

    w = ts.create_w_block(method, npoints, spacing)
    u = ts.create_u_block(method, npoints)
    wu = ts.create_wu_block(method, npoints, spacing)
    for l in range(npoints[0]):
        for r in range(2):
            assert torch.allclose(wu[l,r],
                                  w[l]*u[l,r], atol=1E-10, rtol=1E-5)

    local_logger.debug('pass rel assertion with rel. delta <1E-5')


def test_create_grid1d(level=None):
    local_logger = logger.getChild('test_create_grid1d')
    if level:
        local_logger.setLevel(level)
    origin = 1
    spacing = 0.1
    shape = 10
    offset = (15, 20)
    local_logger.debug(
        f'Testing test_create_grid1d, origin={origin}, '
        f'spacing={spacing}, shape={shape}, '
        f'offset={offset}'
                       )
    grid = ts.create_grid1d(origin, spacing,
                            torch.tensor(offset, dtype=torch.int32),
                            shape)
    for i in range(len(offset)):
        for j in range(shape):
            coord = origin + spacing * offset[i] + spacing * j
            assert torch.allclose(torch.tensor(coord, dtype=grid.dtype),
                                  grid[i,j], atol=1E-10, rtol=1E-5)
    local_logger.debug(
        'pass assertion with rel. delta < 1E-5'
    )

def test_create_node1d_GL(level=None):
    test_create_grid1d(logging.CRITICAL)

    local_logger = logger.getChild('test_create_node1d_GL')
    if level:
        local_logger.setLevel(level)
    origin = 0
    spacing = 2
    shape = 3
    offset = (-1, 1)
    npt = 2
    local_logger.debug(
        f'Testing test_create_node1d_GL,  origin={origin}, '
        f'spacing={spacing}, shape={shape}, '
        f'offset={offset}, npoints={npt}'
                       )
    node1d = ts._create_node1d_GL(npt, origin, spacing,
                                  torch.tensor(offset, dtype=torch.int32),
                                  shape)
    roots, _ = roots_legendre(npt)
    for i in range(len(offset)):
        for j in range(npt):
            for k in range(shape-1):
                x0 = (offset[i] + k) * spacing + origin
                x1 = (offset[i] + k+1) * spacing + origin
                x = (x1-x0) * (roots[j]+1)/2 + x0
                x = torch.tensor(x, dtype=node1d.dtype)
                assert torch.allclose(node1d[i,j,k], x, atol=1E-10, rtol=1E-5)
    local_logger.debug(
        'pass assertion with rel. delta < 1E-5'
    )

def test_create_node1ds(level=None):
    test_create_node1d_GL(logging.CRITICAL)

    local_logger = logger.getChild('test_create_node1ds')
    if level:
        local_logger.setLevel(level)
    origin = (0, 0, 0)
    spacing = (2, 2, 2)
    shape = (3, 3, 3)
    offset = ((-1, 1, 0), (0, 0, 0))
    npt = (2, 3, 4)
    local_logger.debug(
        f'Testing test_create_node1d_GL,  origin={origin}, '
        f'spacing={spacing}, shape={shape}, '
        f'offset={offset}, npoints={npt}'
                       )
    node1ds = ts.create_node1ds('gauss_legendre', npt, origin, spacing,
                                  torch.tensor(offset, dtype=torch.int32),
                                  shape)

    for i in range(len(origin)):
        assert torch.allclose(node1ds[i],
                              ts._create_node1d_GL(npt[i],
                                                   origin[i], spacing[i],
                                                   torch.tensor(offset,
                                                                dtype=torch.int32)[:,i],
                                                   torch.tensor(shape[i],
                                                                dtype=torch.int32))
                              , atol=1E-10, rtol=1E-5)
    local_logger.debug(
        'pass assertion with rel. delta < 1E-5'
    )

def test_eval_qeff(level=None):
    local_logger = logger.getChild('test_eval_qeff')
    if level:
        local_logger.setLevel(level)

    npt = (4, 4, 4)
    method='gauss_legendre'
    origin=(0,0,0)
    grid_spacing=(0.1, 0.1, 0.1)

    Q=[(q,) for q in TEST_Q_VALUES]
    X0=[(0.4,2.4,3.4)] * len(Q)
    X1=[(0.6, 2.6, 3.6)] * len(Q)
    Sigma=[(0.5, 0.5, 0.5)] * len(Q)
    local_logger.debug(
        f'Setup, Q={Q}, X0={X0}, X1={X1}, Sigma={Sigma}, Origin={origin}, GridSpacing={grid_spacing}'
    )
    Q = torch.tensor(Q, dtype=torch.float64)
    X0 = torch.tensor(X0, dtype=torch.float64)
    X1 = torch.tensor(X1, dtype=torch.float64)
    Sigma = torch.tensor(Sigma, dtype=torch.float64)
    offset, shape = ts.compute_charge_box(
        X0, X1, Sigma, THREE_SIGMA, origin, grid_spacing, dtype=torch.float64
    )
    effq2, _ = ts.eval_qeff(Q=Q, X0=X0, X1=X1,
                            Sigma=Sigma, method=method, origin=origin, grid_spacing=grid_spacing,
                            offset=offset, shape=shape, npoints=npt)
    assert_charge_recovery(effq2, Q)

    local_logger.debug('Pass charge-recovery assertion for qline_diff3D')

    # increase grid granularity while keeping the same 3-sigma support box
    grid_spacing2 = (0.01, 0.1, 0.1)
    offset2, shape2 = ts.compute_charge_box(
        X0, X1, Sigma, THREE_SIGMA, origin, grid_spacing2, dtype=torch.float64
    )
    effq3, _ = ts.eval_qeff(Q=Q, X0=X0, X1=X1,
                            Sigma=Sigma, method=method, origin=origin, grid_spacing=grid_spacing2,
                            offset=offset2, shape=shape2, npoints=npt)
    assert_charge_recovery(effq3, Q)
    assert torch.allclose(torch.tensor(effq3.shape[1:]), shape2.to(torch.int64))
    local_logger.debug('Refined-grid charge recovery passes tests.')

def test_compute_qeff(level=None):
    local_logger = logger.getChild('test_eval_qeff')
    if level:
        local_logger.setLevel(level)

    npt = (4, 4, 4)
    method='gauss_legendre'
    origin=(0,0,0)
    grid_spacing=(0.1, 0.1, 0.1)
    Q=[(q,) for q in TEST_Q_VALUES]
    X0=[(0.4,2.4,3.4)] * len(Q)
    X1=[(0.6, 2.6, 3.6)] * len(Q)
    Sigma=[(0.5, 0.5, 0.5)] * len(Q)
    n_sigma = THREE_SIGMA
    Q = torch.tensor(Q, dtype=torch.float64)
    X0 = torch.tensor(X0, dtype=torch.float64)
    X1 = torch.tensor(X1, dtype=torch.float64)
    Sigma = torch.tensor(Sigma, dtype=torch.float64)
    effq, offset = ts.compute_qeff(Q=Q, X0=X0, X1=X1,
                               Sigma=Sigma, n_sigma=n_sigma, method=method,
                               origin=origin, grid_spacing=grid_spacing,
                               npoints=npt)

    assert_charge_recovery(effq, Q)
    local_logger.debug('Passed tests')

    n_sigma = THREE_SIGMA
    effq1, offset = ts.compute_qeff(Q=Q, X0=X0, X1=X1,
                                   Sigma=Sigma, n_sigma=n_sigma, method=method,
                                   origin=origin, grid_spacing=grid_spacing,
                                   npoints=npt,
                                   recenter=True)
    effq2, offset = ts.compute_qeff(Q=Q, X0=X0, X1=X1,
                                   Sigma=Sigma, n_sigma=n_sigma, method=method,
                                   origin=origin, grid_spacing=grid_spacing,
                                   npoints=npt,
                                    skippad=True)
    effq3, offset = ts.compute_qeff(Q=Q, X0=X0, X1=X1,
                                    Sigma=Sigma, n_sigma=n_sigma, method=method,
                                    origin=origin, grid_spacing=grid_spacing,
                                    npoints=npt)

    assert torch.allclose(torch.sum(effq1, dim=(1, 2, 3)), torch.sum(effq2, dim=(1, 2, 3)),
                          atol=1E-4, rtol=1E-5)
    assert torch.allclose(torch.sum(effq1, dim=(1, 2, 3)), torch.sum(effq3, dim=(1, 2, 3)),
                          atol=1E-4, rtol=1E-5)
    logger.debug('The relative difference when setting 4 sigma band is roughly 1E-4')

def main():
    print('------ test_QModel ------')
    test_QModel()

    print('-------- test_roots_legendre ---------')
    test_roots_legendre()

    print('-------- test_create_w1d_GL() ----------')
    test_create_w1d_GL()

    print('-------- test_create_w1ds() ----------')
    test_create_w1ds()

    print('-------- test_create_w_block() --------')
    test_create_w_block()

    print('-------- test_create_u1d_GL() --------')
    test_create_u1d_GL()

    print('-------- test_create_u1ds() ---------')
    test_create_u1ds()

    print('-------- test_create_u_block ---------')
    test_create_u_block()

    print('-------- test_create_wu_block ---------')
    test_create_wu_block()

    print('-------- test_create_grid1d ---------')
    test_create_grid1d()

    print('-------- test_create_node1d_GL ---------')
    test_create_node1d_GL()

    print('-------- test_create_node1ds ---------')
    test_create_node1ds()

    print('-------- test_eval_qeff ---------')
    test_eval_qeff(level=None)

    print('-------- test_compute_qeff ---------')
    test_compute_qeff(level=None)


if __name__ == '__main__':
    try:
        opt = sys.argv[1]
        if opt.lower() == 'debug':
            logging.basicConfig(level=logging.DEBUG)
        elif opt.lower() == 'warning':
            logging.basicConfig(level=logging.WARNING)
        elif opt.lower() == 'info':
            logging.basicConfig(level=logging.INFO)
        else:
            print('Usage: test_grid.py [debug|warning|info]')
            exit(-1)
    except IndexError:
        # logging.basicConfig(level=logging.DEBUG)
        print('To use system default logging level')

    main()
