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

def test_create_wblock(level=None):
    test_create_w1ds(logging.CRITICAL)

    local_logger = logger.getChild('test_create_wblock')
    if level:
        local_logger.setLevel(level)
    npoints = (3, 2, 1)
    spacing = (2., 2., 2.)
    local_logger.debug(
        f'Testing {npoints}-point weight block calculation in {len(npoints)}D,'
        f' interval width == {spacing}'
    )
    wblock  = ts.create_wblock('gauss_legendre', npoints, spacing)
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
    wblock  = ts.create_wblock('gauss_legendre', npoints, spacing)
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
    wblock  = ts.create_wblock('gauss_legendre', npoints, spacing)
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

def test_create_ublock(level=None):
    test_create_u1ds(logging.CRITICAL)

    local_logger = logger.getChild('test_create_ublock')
    if level:
        local_logger.setLevel(level)

    npoints = (3, 2, 1)
    local_logger.debug(
        f'Testing u block calculation in {len(npoints)}D on {npoints}-point GL'
    )
    ublock  = ts.create_ublock('gauss_legendre', npoints)
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
    ublock  = ts.create_ublock('gauss_legendre', npoints)
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

    ublock  = ts.create_ublock('gauss_legendre', npoints)
    u1ds = ts._create_u1ds('gauss_legendre', npoints)
    for i in range(npoints[0]):
        assert torch.allclose(u1ds[0][i],
                              ublock[i], atol=1E-10, rtol=1E-5)

    local_logger.debug('pass rel assertion with rel. delta <1E-5')


def main():
    print('------ test_QModel ------')
    test_QModel()

    print('-------- test_roots_legendre ---------')
    test_roots_legendre()

    print('-------- test_create_w1d_GL() ----------')
    test_create_w1d_GL()

    print('-------- test_create_w1ds() ----------')
    test_create_w1ds()

    print('-------- test_create_wblock() --------')
    test_create_wblock()

    print('-------- test_create_u1d_GL() --------')
    test_create_u1d_GL()

    print('-------- test_create_u1ds() ---------')
    test_create_u1ds()

    print('-------- test_create_ublock ---------')
    test_create_ublock()

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
