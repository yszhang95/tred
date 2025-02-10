import tred.raster.steps as ts
from tred.raster.steps import (
    compute_index, compute_coordinate, compute_charge_box,
    qline_diff3D, roots_legendre,
)

import json
import logging
import sys

import torch

logger = logging.getLogger('tred/tests/effq/test_effq.py')

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
    with open('exact_qline_gaus.json') as f:
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

    sp_u = (sp.special.roots_legendre(npt)[0]+1)/2
    sp_u = torch.tensor(sp_u, dtype=torch.float32)
    u1d = ts._create_u1d_GL(npt)
    assert torch.allclose(u1d[:,1], sp_u, atol=1E-10, rtol=1E-5)
    assert torch.allclose(u1d[:,0], 1-sp_u, atol=1E-10, rtol=1E-5)
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
            assert torch.allclose(torch.tensor(coord),
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
                x = torch.tensor(x, dtype=torch.float32)
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

def prod_xcub_ycub_zcub(Q, X0, X1, Sigma, x, y, z):
    local_logger = logger.getChild('prod_xcub_ycub_zcub=x^3*y^3*z^3')
    local_logger.debug('Ignoring Q, X0, X1, Sigma')
    output = x**3 * y**3 * z**3
    return output

def prod_xcub_ycub_zcub_margin0(Q, X0, X1, Sigma, x, y, z):
    local_logger = logger.getChild('prod_xcub_ycub_zcub=x^3*y^3*z^3, margins are 0')
    local_logger.debug('Ignoring Q, X0, X1, Sigma')
    output = x**3 * y**3 * z**3
    output[...,0,:,:] = 0
    output[...,-1,:,:] = 0
    output[...,0,:] = 0
    output[...,-1,:] = 0
    output[...,0] = 0
    output[...,-1] = 0
    return output

def test_eval_qmodel(level=None):
    local_logger = logger.getChild('test_eval_qmodel')
    if level:
        local_logger.setLevel(level)

    # npt = (4, 4, 4)
    npt = (2, 2, 2)
    method='gauss_legendre'
    origin=(0,0,0)
    grid_spacing=(0.1, 0.1, 0.1)
    offset=[(0,20,30)]
    shape=(10, 10, 10)

    local_logger.debug(f'Testing {npt}-point GL rule using {repr(prod_xcub_ycub_zcub)};'
                       ' Contributions from Q, X0, X1, Sigma, are ignored')

    x, y, z = [ts.create_grid1d(orig, spacing, off, shp+1) for orig, spacing, off, shp in zip(origin, grid_spacing, torch.tensor(offset).T, shape)]
    x0, x1 = x[0,0], x[0,-1]
    y0, y1 = y[0,0], y[0,-1]
    z0, z1 = z[0,0], z[0,-1]
    local_logger.debug(f'Setup, x in [{x0},{x1}], '
                       f'y in [{y0},{y1}], z in [{z0},{z1}]')

    x, y, z = ts.create_node1ds(method, npt, origin, grid_spacing, torch.tensor(offset), shape)
    q = ts.eval_qmodel(None, None, None, None, x, y, z, qmodel=prod_xcub_ycub_zcub)
    w = ts.create_w_block(method, npt, grid_spacing)

    effq = w[None,...,None,None,None]*q

    # effq = qeff.create_qeff(None, dummy, None, None, qmodel=mymodel)
    msg = f'output shape {q.shape}, sum of fn values at nodes {torch.sum(q).item()}. Preknown value for integral of (xyz)^3 is 177.7344'
    local_logger.debug(msg)
    assert abs(torch.sum(effq).item() - 177.7344)/177.7344 < 1E-5, msg
    local_logger.debug('Pass assertion for x^3 * y^3 * t^3 at relative delta 1E-5')

def test_eval_qeff(level=None):
    local_logger = logger.getChild('test_eval_qeff')
    if level:
        local_logger.setLevel(level)

    npt = (4, 4, 4)
    # npt = (2, 2, 2)
    method='gauss_legendre'
    origin=(0,0,0)
    grid_spacing=(0.1, 0.1, 0.1)
    offset=[(0,20,30)]
    shape=(10, 10, 10)

    local_logger.debug(f'Testing {npt}-point GL rule using {repr(prod_xcub_ycub_zcub)};'
                       ' Contributions from Q, X0, X1, Sigma, are ignored')

    x, y, z = [ts.create_grid1d(orig, spacing, off, shp+1) for orig, spacing, off, shp in zip(origin, grid_spacing, torch.tensor(offset).T, shape)]
    x0, x1 = x[0,0], x[0,-1]
    y0, y1 = y[0,0], y[0,-1]
    z0, z1 = z[0,0], z[0,-1]
    local_logger.debug(f'Setup, x in [{x0},{x1}], '
                       f'y in [{y0},{y1}], z in [{z0},{z1}]')

    dummyQ = torch.arange(1).view(len(offset),-1)
    dummyX0 = torch.arange(3).view(len(offset),-1)
    dummyX1 = torch.arange(3).view(len(offset),-1)
    dummySigma = torch.arange(3).view(len(offset),-1)

    effq = ts.eval_qeff(dummyQ, dummyX0, dummyX1, dummySigma,
                     torch.tensor(offset), torch.tensor(shape), origin, grid_spacing, method, npt,
                     qmodel=prod_xcub_ycub_zcub)

    msg = f'output shape {effq.shape}, sum of fn values at nodes {torch.sum(effq).item()}. Preknown value for integral of (xyz)^3 is 177.7344'
    local_logger.debug(msg)
    assert abs(torch.sum(effq).item() - 177.7344)/177.7344 < 1E-5, msg
    local_logger.debug('Pass assertion for x^3 * y^3 * t^3 at relative delta 1E-5')

    # test skipad
    effq = ts.eval_qeff(dummyQ, dummyX0, dummyX1, dummySigma,
                           torch.tensor(offset)-1, torch.tensor(shape)+2, origin, grid_spacing, method, npt,
                           qmodel=prod_xcub_ycub_zcub_margin0, skippad=True)
    msg = f'output shape {effq.shape}, sum of fn values at nodes {torch.sum(effq).item()}. Preknown value for integral of (xyz)^3 is 177.7344'
    local_logger.debug(msg)
    assert abs(torch.sum(effq).item() - 177.7344)/177.7344 < 1E-5, msg
    local_logger.debug('Pass assertion for x^3 * y^3 * t^3 at relative delta 1E-5, skippad=True')

    ilinear = lambda x, y, t : x * y * t
    x = torch.linspace(0, 1, 11)
    y = torch.linspace(2, 3, 11)
    t = torch.linspace(3, 4, 11)
    xgrid, ygrid, tgrid = torch.meshgrid(x, y, t, indexing='ij')
    I = ilinear(xgrid, ygrid, tgrid)
    Y = effq
    assert abs((torch.sum(Y * I) - 1318.3282).item())/1318.3282 <1E-5
    local_logger.debug('Pass assertion for x^4 * y^4 * t^4 after multiplying linear model x*y*t')

    # # Asserted

    Q=(1,)
    X0=[(0.4,2.4,3.4)]
    X1=[(0.6, 2.6, 3.6)]
    Sigma=[(0.5, 0.5, 0.5)]
    local_logger.debug(
        f'Setup, Q={Q}, X0={X0}, X1={X1}, Sigma={Sigma}, Origin={origin}, GridSpacing={grid_spacing}, Offset={offset}, Shape={shape}'
    )
    Q = torch.tensor(Q)
    X0 = torch.tensor(X0)
    X1 = torch.tensor(X1)
    Sigma = torch.tensor(Sigma)
    effq2 = ts.eval_qeff(Q=Q, X0=X0, X1=X1,
                            Sigma=Sigma, method=method, origin=origin, grid_spacing=grid_spacing,
                            offset=torch.tensor(offset), shape=torch.tensor(shape), npoints=npt)
    # print('Sum of Line conv Gaus', torch.sum(effq2))
    qint = 0.3137
    msg = f'Sum of qline_diff3D {torch.sum(effq2)}, predefined value for comparison {qint}'
    assert torch.isclose(torch.sum(effq2), torch.tensor([qint,]), atol=1E-4, rtol=1E-5), msg

    local_logger.debug('Pass assertion for qline_diff3D')

    # test xyzlimit and shapelimit
    # increase grid granularity
    grid_spacing2 = (0.001, 0.1, 0.1)
    shape2 = (1000, 10, 10)
    offset2 = (0, 20, 30)
    shape_limit = 10000000000
    effq3 = ts.eval_qeff(Q=Q, X0=X0, X1=X1,
                            Sigma=Sigma, method=method, origin=origin, grid_spacing=grid_spacing2,
                            offset=torch.tensor(offset), shape=torch.tensor(shape2), npoints=npt,
                            shape_limit=torch.tensor(shape_limit))
    msg = f'Sum of qline_diff3D {torch.sum(effq3)}, predefined value for comparison {qint}'
    assert torch.isclose(torch.sum(effq3), torch.tensor([qint,]), atol=1E-4, rtol=1E-5), msg

    shape_limit = 100
    effq3 = ts.eval_qeff(Q=Q, X0=X0, X1=X1,
                            Sigma=Sigma, method=method, origin=origin, grid_spacing=grid_spacing2,
                            offset=torch.tensor(offset), shape=torch.tensor(shape2), npoints=npt,
                            shape_limit=torch.tensor(shape_limit))
    msg = f'Sum of qline_diff3D {torch.sum(effq3)}, predefined value for comparison {qint}'
    assert torch.isclose(torch.sum(effq3), torch.tensor([qint,]), atol=1E-4, rtol=1E-5), msg

    grid_spacing4 = (0.1, 0.001, 0.1)
    shape4 = (10, 1000, 10)
    offset4 = [(0, 2000, 30),]
    shape_limit4 = 100
    effq4 = ts.eval_qeff(Q=Q, X0=X0, X1=X1,
                            Sigma=Sigma, method=method, origin=origin, grid_spacing=grid_spacing4,
                            offset=torch.tensor(offset4), shape=torch.tensor(shape4), npoints=npt,
                            shape_limit=torch.tensor(shape_limit))
    msg = f'Sum of qline_diff3D {torch.sum(effq4)}, predefined value for comparison {qint}'
    assert torch.isclose(torch.sum(effq4), torch.tensor([qint,]), atol=1E-4, rtol=1E-5), msg

    grid_spacing5 = (0.1, 0.1, 0.001)
    shape5 = (10, 10, 1000)
    offset5 = [(0, 20, 3000),]
    shape_limit5 = 100
    effq5 = ts.eval_qeff(Q=Q, X0=X0, X1=X1,
                            Sigma=Sigma, method=method, origin=origin, grid_spacing=grid_spacing5,
                            offset=torch.tensor(offset5), shape=torch.tensor(shape5), npoints=npt,
                            shape_limit=torch.tensor(shape_limit))
    msg = f'Sum of qline_diff3D {torch.sum(effq5)}, predefined value for comparison {qint}'
    assert torch.isclose(torch.sum(effq5), torch.tensor([qint,]), atol=1E-4, rtol=1E-5), msg

    grid_spacing6 = (0.1, 0.005, 0.001)
    shape6 = (10, 200, 1000)
    offset6 = [(0, 400, 3000),]
    shape_limit6 = 100
    effq6 = ts.eval_qeff(Q=Q, X0=X0, X1=X1,
                            Sigma=Sigma, method=method, origin=origin, grid_spacing=grid_spacing6,
                            offset=torch.tensor(offset6), shape=torch.tensor(shape6), npoints=npt,
                            shape_limit=torch.tensor(shape_limit))
    msg = f'Sum of qline_diff3D {torch.sum(effq6)}, predefined value for comparison {qint}'
    assert torch.isclose(torch.sum(effq6), torch.tensor([qint,]), atol=1E-4, rtol=1E-5), msg
    local_logger.debug('Chunking on x, y, z passes tests.')

    # test mem_limit
    batch_size = 20_000
    Q=(1,) * batch_size
    X0=[(0.4,2.4,3.4)] * batch_size
    X1=[(0.6, 2.6, 3.6)] * batch_size
    Sigma=[(0.5, 0.5, 0.5)] * batch_size
    npt = (4, 4, 4)
    # npt = (2, 2, 2)
    method='gauss_legendre'
    origin=(0,0,0)
    grid_spacing=(0.1, 0.1, 0.1)
    offset=[(0,20,30)] *batch_size
    shape=(10, 10, 10)

    torch.cuda.reset_peak_memory_stats()
    Q = torch.tensor(Q).to('cuda')
    X0 = torch.tensor(X0).to('cuda')
    X1 = torch.tensor(X1).to('cuda')
    Sigma = torch.tensor(Sigma).to('cuda')
    mem_limit = 1024 # MB
    effq7 = ts.eval_qeff(Q=Q, X0=X0, X1=X1,
                            Sigma=Sigma, method=method, origin=origin, grid_spacing=grid_spacing,
                            offset=torch.tensor(offset, device='cuda'),
                            shape=torch.tensor(shape, device='cuda'), npoints=npt,
                            mem_limit = mem_limit)
    msg = f'Sum of qline_diff3D {torch.sum(effq7)}, predefined value for comparison {qint}'
    assert torch.allclose(torch.sum(effq7, dim=(1,2,3)), torch.tensor([qint,],device='cuda').repeat(batch_size), atol=1E-4, rtol=1E-5), msg
    msg = f'Peak cuda memory {torch.cuda.max_memory_allocated()/1024**2} MB, soft memory limit is {mem_limit} MB'
    assert torch.cuda.max_memory_allocated() / 1024**2 < mem_limit, msg
    local_logger.debug('Passed the test. Peak memory is under control.')


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

    print('-------- test_eval_qmodel ---------')
    test_eval_qmodel(level=None)

    print('-------- test_eval_qeff ---------')
    test_eval_qeff(level=None)

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
