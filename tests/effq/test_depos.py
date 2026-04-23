import pytest
import torch

from tred.raster.depos import binned_1d, binned_nd
from tred.types import index_dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_binned_1d_gaussian(dtype):
    centers = torch.tensor([1.0], dtype=torch.float32)
    widths = torch.tensor([0.2], dtype=torch.float32)
    charges = torch.tensor([2.0], dtype=torch.float32)

    qeff, offset = binned_1d(0.5, centers, widths, charges, dtype=dtype)

    assert qeff.shape[0] == centers.shape[0]
    assert qeff.dtype == dtype
    assert offset.dtype == index_dtype
    assert torch.isfinite(qeff).all()
    assert qeff.sum() > 0
    assert qeff.sum() <= charges.to(dtype=dtype).sum()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_binned_1d_gaussian_numerical(dtype):
    centers = torch.tensor([1.0], dtype=torch.float32)
    widths = torch.tensor([0.2], dtype=torch.float32)
    charges = torch.tensor([2.0], dtype=torch.float32)

    qeff, offset = binned_1d(0.5, centers, widths, charges, dtype=dtype)
    expected = torch.tensor(
        [[1.1372730579495283e-07,
          0.077099762421785289,
          1.8458002477018178,
          0.077099762421785289]],
        dtype=dtype,
    )

    assert offset.equal(torch.tensor(0, dtype=index_dtype))
    assert torch.allclose(qeff, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_binned_1d_zero_width_spike(dtype):
    centers = torch.tensor([1.0], dtype=torch.float32)
    widths = torch.tensor([0.0], dtype=torch.float32)
    charges = torch.tensor([2.0], dtype=torch.float32)

    qeff, offset = binned_1d(0.5, centers, widths, charges, dtype=dtype)
    expected_grid_index = torch.round(centers / 0.5).to(index_dtype)
    rel_index = (expected_grid_index - offset).to(dtype=torch.long)

    assert qeff.dtype == dtype
    assert offset.dtype == index_dtype
    assert torch.isfinite(qeff).all()
    assert torch.isclose(qeff.sum(), charges.to(dtype=dtype).sum())
    assert qeff[0, rel_index.item()] == charges.to(dtype=dtype)[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_binned_nd_zero_width_spike_numerical(dtype):
    centers = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    sigmas = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    charges = torch.tensor([3.0], dtype=torch.float32)

    qeff, offset = binned_nd((0.5, 0.5), centers, sigmas, charges, dtype=dtype)
    expected = torch.tensor(
        [[[0.0, 0.0],
          [0.0, 3.0]]],
        dtype=dtype,
    )

    assert offset.equal(torch.tensor([[1, 1]], dtype=index_dtype))
    assert torch.allclose(qeff, expected)
