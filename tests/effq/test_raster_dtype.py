import pytest
import torch

from tred.graph import Raster
from tred.raster.depos import binned_1d, binned_nd
from tred.raster.steps import compute_qeff
from tred.types import index_dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_binned_1d_dtype(dtype):
    centers = torch.tensor([1.0], dtype=torch.float32)
    widths = torch.tensor([0.2], dtype=torch.float32)
    charges = torch.tensor([2.0], dtype=torch.float32)

    qeff, offset = binned_1d(0.5, centers, widths, charges, dtype=dtype)

    assert qeff.dtype == dtype
    assert offset.dtype == index_dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_binned_nd_dtype(dtype):
    centers = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    sigmas = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    charges = torch.tensor([1.0], dtype=torch.float32)

    qeff, offset = binned_nd((0.5, 0.5, 0.5), centers, sigmas, charges, dtype=dtype)

    assert qeff.dtype == dtype
    assert offset.dtype == index_dtype
    assert torch.isfinite(qeff).all()
    assert qeff.sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compute_qeff_dtype(dtype):
    Q = torch.tensor([(1,)], dtype=torch.float32)
    X0 = torch.tensor([(0.4, 2.4, 3.4)], dtype=torch.float32)
    X1 = torch.tensor([(0.6, 2.6, 3.6)], dtype=torch.float32)
    Sigma = torch.tensor([(0.5, 0.5, 0.5)], dtype=torch.float32)

    qeff, offset = compute_qeff(
        Q=Q,
        X0=X0,
        X1=X1,
        Sigma=Sigma,
        n_sigma=(0.7, 0.7, 0.7),
        origin=(0, 0, 0),
        grid_spacing=(0.1, 0.1, 0.1),
        method='gauss_legendre',
        npoints=(2, 2, 2),
        dtype=dtype,
    )

    assert qeff.dtype == dtype
    assert offset.dtype == index_dtype
    assert torch.isfinite(qeff).all()
    assert qeff.sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_raster_forward_dtype(dtype):
    raster = Raster(
        velocity=0.1,
        grid_spacing=(0.1, 0.1, 0.1),
        pdims=(1, 2),
        tdim=-1,
        dtype=dtype,
    )
    sigma = torch.tensor([(0.05, 0.5, 0.5)], dtype=torch.float32)
    time = torch.tensor([0.4], dtype=torch.float32)
    charge = torch.tensor([1.0], dtype=torch.float32)
    tail = torch.tensor([(0.06, 2.4, 3.4)], dtype=torch.float32)
    head = torch.tensor([(0.04, 2.6, 3.6)], dtype=torch.float32)

    depo = raster(sigma, time, charge, tail)
    step = raster(sigma, time, charge, tail, head)

    for block in (depo, step):
        assert block.data.dtype == dtype
        assert block.location.dtype == index_dtype
        assert torch.isfinite(block.data).all()
        assert block.data.sum() > 0
