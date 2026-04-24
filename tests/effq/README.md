# Raster Backend Notes

This note summarizes the current state of:

- `src/tred/raster/steps.py`
- `src/tred/raster/depos.py`
- `src/tred/graph.py` (`Raster`)
- related tests under `tests/effq/`

## Scope

The known tensor-layout difference was excluded from review:

- some code paths use `(vdim, N)`
- others use `(N, vdim)`

That mismatch is already known and is not repeated below.

## Source Summary

### `src/tred/raster/steps.py`

This module is the step-rasterization backend. It provides:

- coordinate/index conversion: `compute_coordinate()`, `compute_index()`
- step bounds and charge-box helpers: `compute_bounds_X0X1()`, `compute_bounds_X0_X1()`, `compute_charge_box()`
- Gauss-Legendre helpers for weights, nodes, and interpolation
- line-like and point-like charge models: `qline_diff3D()`, `qpoint_diff3D()`
- main evaluation entry points: `eval_qeff()` and `compute_qeff()`

In practice, this is the implementation used by `tred.graph.raster_steps()` and
by the step path of `Raster.forward()`.

### `src/tred/raster/depos.py`

This module is the point-deposition rasterization backend. It provides:

- `binned_1d()` for 1D Gaussian depos
- `binned_nd()` for N-D Gaussian depos
- `binned()` as the public wrapper

In practice, this is the implementation used by the depo path of
`Raster.forward()`.

### `src/tred/graph.py` (`Raster`)

`Raster` is the graph wrapper around the two raster backends.

- `_head_time_offset_from_tail()` computes the offset added to tail time to get
  head time for steps
- `_transform()` permutes axes into raster coordinates and optionally replaces
  the drift axis with time
- `forward()` dispatches:
  - to `raster_depos()` when `head is None`
  - to `raster_steps()` when `head is not None`

## Current Source Status

### `Raster._head_time_offset_from_tail()`

The old `Raster._time_diff()` helper has been renamed to
`Raster._head_time_offset_from_tail()` and now documents that it returns the
offset added to tail time to get head time.

The implementation and tests use:

```python
(tail - head) / velocity
```

The old `_time_diff()` docstring was stale. The source and tests now agree.

### Raster tail-time convention

`Raster.forward()` treats the input `time` as the tail time. For steps, it
derives the head time as:

```python
head_time = time + (tail - head) / velocity
```

This matches the Drifter convention:

- `Drifter._order_step_endpoints_for_drift_time()` selects the reordered tail
  endpoint
- `Drifter.forward()` computes `dtime` from that reordered tail
- `Raster.forward()` receives that `dtime` as its `time` input

### `Raster.forward()` sigma transform

`sigma` is transformed before dispatching to either the depo or step raster
backend.

That common-path transform is required because `Raster` works in raster
coordinates, and the drift-axis width must be converted from distance-width to
time-width for both:

- depos
- steps

This is covered by `tests/effq/test_raster_dtype.py`.

### Explicit raster compute dtype

Raster now supports an explicit compute dtype with default `torch.float64`.

Current behavior:

- float inputs may remain `float32`
- raster compute may be requested as either:
  - `torch.float32`
  - `torch.float64`
- index-like outputs remain `int32`

This is wired through:

- `tred.graph.Raster`
- `tred.graph.raster_steps()`
- `tred.raster.steps.compute_qeff()`
- `tred.raster.steps.eval_qeff()`
- `tred.raster.depos.binned()`
- `tred.raster.depos.binned_nd()`
- `tred.raster.depos.binned_1d()`

### `binned_1d()` and `binned_nd()`

The earlier indexing-site issue with `int32` advanced indexing is fixed.

The scalar-grid branch in `binned_nd()` is also fixed to use the raster compute
dtype instead of the index dtype.

The current tests also cover the finite-width `vdim=1` consistency between
`binned_nd()` and `binned_1d()`.

The non-integer support-width regression behind issue `#55` is now covered in
both the 1D and true 2D finite-width paths. The current implementation uses
`ceil()` when converting `nsigma * sigma / grid` into a half-width bin count,
which correctly extends the support window instead of truncating it.

### `eval_qeff()` tests and current charge checks

`tests/effq/test_effq.py` now calls `eval_qeff()` using the current interface:

- `qline_model`
- `qpoint_model`

The older `qmodel=...` call path is no longer used there.

The older hard-coded `0.3137` / `0.313723` sums are no longer treated as
general test targets. The direct `qeff` tests now:

- use `+-3 sigma` support boxes
- compare recovered rasterized charge against the input `Q`
- include direct `qpoint` coverage, including an `erf`-based finite-box check
  for `qpoint_diff3D()`

### Dtype-dependent boundary behavior

The reviewed exact-on-grid-edge charge-box case is now dtype-stable across
`float32` and `float64`.

The sensitive path was:

- `compute_index()`
- `compute_charge_box()`

The fix computes the index ratio in `float64`, snaps values within a
source-dtype tolerance of an integer boundary to that integer, and only then
applies `floor()`.

This is covered by:

- `tests/effq/test_grid.py::test_compute_charge_box_dtype_stable_on_grid_boundary`

## Test Review

### `tests/effq/test_grid.py`

This file covers:

- `compute_index()`
- `compute_coordinate()`
- `_stack_X0X1()`
- `compute_bounds_X0X1()`
- `compute_bounds_X0_X1()`
- `compute_charge_box()`
- `reduce_to_universal()`

Current status:

- the stale `float32` expected coordinates have been updated for the reviewed
  case
- the strict floating-point bound coverage checks were replaced with a small
  tolerance helper
- that helper explicitly notes that it is only intended for the current
  grid-aligned cases and is not a general rule for arbitrary decimal bounds
- the focused regression test for dtype-stable charge-box indexing on a grid
  boundary remains in place

### `tests/effq/test_effq.py`

This file exercises:

- quadrature helpers
- interpolation helpers
- `eval_qeff()`
- `compute_qeff()`

Current status:

- the `exact_qline_gaus.json` path issue is fixed by resolving it relative to
  the test file
- the older `qmodel=...` call against `eval_qeff()` has been removed
- the old hard-coded `qeff` integral checks have been replaced with `+-3 sigma`
  charge-recovery checks against `Q`
- `qpoint` coverage includes:
  - an exact finite-box `erf` comparison for `qpoint_diff3D()`
  - a point-branch `eval_qeff()` charge-recovery test
  - a short-segment agreement test between forced `qline` and forced `qpoint`
- the helper tests are aligned with the current `float64` default

### `tests/effq/test_raster_steps.py`

This file covers:

- `Raster._transform()`
- `Raster._head_time_offset_from_tail()`
- `tred.graph.raster_steps()`
- a step-based `Raster` integration check
- one plotting helper

Current status:

- `_head_time_offset_from_tail()` tests confirm the implementation returns
  `(tail - head) / velocity`
- `_transform()` tests remain useful
- the stale integral checks have been replaced with charge-recovery assertions
- `test_drift_raster_positions()` is still a plotting script rather than a
  strong unit test

### `tests/effq/test_raster_dtype.py`

This tracked smoke test covers:

- `binned_nd(..., dtype=...)`
- `compute_qeff(..., dtype=...)`
- `Raster.forward(..., dtype=...)`

for both:

- `torch.float32`
- `torch.float64`

It checks:

- output floating dtype follows the requested compute dtype
- offsets remain `int32`
- outputs are finite and non-zero

This is a focused smoke test for the dtype contract. It does not replace the
older numerical/reference tests.

### `tests/effq/test_depos.py`

This file now provides direct depos backend tests.

It covers:

- `binned_1d()` finite-width Gaussian behavior
- `binned_1d()` finite-width numerical regression
- `binned_1d()` zero-width spike behavior
- `binned_nd()` finite-width `vdim=1` consistency with `binned_1d()`
- `binned_nd()` finite-width 2D numerical regression for non-integer `sigma/grid`
- `binned_nd()` zero-width spike numerical regression

for both:

- `torch.float32`
- `torch.float64`

`minbins` and the `binned()` wrapper remain reasonable follow-ups.

## Commands Recently Used

The current effq test suite was checked with:

```bash
PYTHONPATH=src pytest -q tests/effq
```

Observed result in this tree:

- `45 passed`

The focused dtype smoke test was checked with:

```bash
PYTHONPATH=src pytest -q tests/effq/test_raster_dtype.py
```

Observed result in this tree:

- `8 passed`

The focused depos backend tests were checked with:

```bash
PYTHONPATH=src pytest -q tests/effq/test_depos.py
```

Observed result in this tree:

- `12 passed`
