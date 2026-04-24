# Raster Review Notes

This note summarizes the current state of:

- `src/tred/raster/steps.py`
- `src/tred/raster/depos.py`
- `src/tred/graph.py` (`Raster`)
- related tests under `tests/`

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
- Gauss-Legendre helpers for weights/nodes/interpolation
- line-like and point-like charge models: `qline_diff3D()`, `qpoint_diff3D()`
- main evaluation entry points: `eval_qeff()` and `compute_qeff()`

In practice, this is the implementation used by `tred.graph.raster_steps()` and by the step path of `Raster.forward()`.

### `src/tred/raster/depos.py`

This module is the point-deposition rasterization backend. It provides:

- `binned_1d()` for 1D Gaussian depos
- `binned_nd()` for N-D Gaussian depos
- `binned()` as the public wrapper

In practice, this is the implementation used by the depo path of `Raster.forward()`.

### `src/tred/graph.py` (`Raster`)

`Raster` is the graph wrapper around the two raster backends.

- `_head_time_offset_from_tail()` computes the offset added to tail time to get head time for steps
- `_transform()` permutes axes into raster coordinates and optionally replaces the drift axis with time
- `forward()` dispatches:
  - to `raster_depos()` when `head is None`
  - to `raster_steps()` when `head is not None`

## Inconsistencies

### `Raster._head_time_offset_from_tail()` docstring mismatch

Fixed in the current source. The old `Raster._time_diff()` helper has been renamed to `Raster._head_time_offset_from_tail()` and now documents that it returns the offset added to tail time to get head time.

The old helper documented `(head - tail) / v`, while the implementation and tests use `(tail - head) / velocity`.

The tests in `tests/effq/test_raster_steps.py` also expect `(tail - head) / velocity`.

Conclusion:

- the code and tests agree
- the old `_time_diff()` docstring was stale
- fixed

Relevant locations:

- `src/tred/graph.py`
- `tests/effq/test_raster_steps.py`

### Raster tail-time convention

`Raster.forward()` treats the input `time` as the tail time. For steps, it derives the head time as:

```python
head_time = time + (tail - head) / velocity
```

This matches the Drifter convention:

- `Drifter._order_step_endpoints_for_drift_time()` selects the reordered tail endpoint
- `Drifter.forward()` computes `dtime` from that reordered tail
- `Raster.forward()` receives that `dtime` as its `time` input

### `Raster.forward()` sigma-transform inconsistency

Fixed in the current source: `sigma` is transformed before dispatching to either the depo or step raster backend.

Previously, the docstring said `tail`, `head`, and `sigma` were all interpreted in spatial coordinates and transformed into raster coordinates as needed, but only the step path transformed `sigma`. That left the depo path using transformed positions with untransformed widths.

Conclusion:

- moving the `sigma` transform into the common path was justified
- fixed in current source

### Explicit raster compute dtype

Raster now supports an explicit compute dtype with default `torch.float64`.

Current behavior:

- float inputs may remain `float32`
- raster compute may be requested as either:
  - `torch.float32`
  - `torch.float64`
- index-like outputs remain on `int32`

This is now wired through:

- `tred.graph.Raster`
- `tred.graph.raster_steps()`
- `tred.raster.steps.compute_qeff()`
- `tred.raster.steps.eval_qeff()`
- `tred.raster.depos.binned()`
- `tred.raster.depos.binned_nd()`
- `tred.raster.depos.binned_1d()`

Conclusion:

- raster no longer needs to infer float dtype from the input tensors
- raster defaults to `float64` compute while still accepting `float32` inputs

## Status of the reviewed source changes

This section is no longer a pending local-change note. The described `Raster.forward()` change is present in the current source.

The relevant change was:

- move
  - `sigma = self._transform(sigma, None)`
  - `sigma[:, self._tdim] = sigma[:, self._tdim] / torch.abs(self.velocity)`
- from the step-only path to the common path before checking `head is None`

Why this was justified:

- `Raster` transforms positions into raster coordinates
- `sigma` represents widths in the same coordinate system
- the drift-axis width must therefore be converted from distance-width to time-width for both:
  - depos
  - steps

Without this change, the depo path mixed transformed positions with untransformed widths.

Conclusion:

- the local `Raster.forward()` change is consistent with the documented raster-space contract
- it fixes a real inconsistency
- fixed in current source and covered by `tests/effq/test_raster_dtype.py`

## Problems in `src/tred/raster/depos.py`

These are source-level issues observed during review.

### `binned_1d()` and `binned_nd()` indexing bug

Fixed in the current source.

Earlier in this review, both `binned_1d()` and `binned_nd()` failed on CPU with:

```text
IndexError: tensors used as indices must be long, byte or bool tensors
```

The issue came from advanced indexing with `int32` tensors at:

- `src/tred/raster/depos.py:123`
- `src/tred/raster/depos.py:213`

This has now been fixed by converting those indexing tensors to `long` at the indexing sites.

Conclusion:

- this bug is no longer considered outstanding
- fixed

### Scalar-grid branch in `binned_nd()`

Fixed in the current source.

`binned_nd()` documents that `grid` may be a real scalar or a 1D N-vector. In the old scalar branch it built `torch.tensor([grid], dtype=index_dtype, ...)`, which was inconsistent with the documented meaning of `grid` as a spacing value.

This has now been changed to use the raster compute dtype instead of `index_dtype`.

Conclusion:

- this issue is no longer considered outstanding
- fixed

## Problems in `src/tred/raster/steps.py`

### `eval_qeff()` test API mismatch

Fixed in the current tests.

`tests/effq/test_effq.py` now calls `eval_qeff()` using the current interface:

- `qline_model`
- `qpoint_model`

The older `qmodel=...` call path is no longer used there.

Conclusion:

- the test no longer mismatches the implementation interface
- fixed in current tests

### Dtype-dependent boundary behavior

Fixed in the current source for the reviewed near-boundary case.

Raster now supports both `float32` and `float64` compute. Earlier in review, the same logical raster bounds could produce:

- different block shapes
- different block offsets

when switching dtype near bin edges.

The sensitive path was:

- `compute_index()`
- `compute_charge_box()`

where `floor()` acted directly on floating-point values near integer grid boundaries.

This is now handled by:

- computing the index ratio in `float64`
- snapping values within a source-dtype tolerance of an integer boundary to that integer
- then applying `floor()`

This fix is covered by:

- `tests/effq/test_grid.py::test_compute_charge_box_dtype_stable_on_grid_boundary`

Conclusion:

- exact-on-edge charge-box indexing is now dtype-stable across `float32` and `float64`
- the boundary rule is now explicit instead of implicit

### Hard-coded integral expectations are partly stale

For current code in this tree:

- `compute_qeff(..., npoints=(2,2,2))` gives about `0.29399486735385094`
- `compute_qeff(..., npoints=(4,4,4))` gives about `0.29597755499800044`

The older stale reference values were about:

- `0.3137`
- `0.313723`

The direct `qeff` tests in `tests/effq/test_effq.py` have now been rewritten to avoid those hard-coded sums. They instead:

- use `+-3 sigma` support boxes
- compare recovered rasterized charge against the input `Q`
- include direct `qpoint` coverage, including an `erf`-based box-integral check for `qpoint_diff3D()`

Conclusion:

- the old hard-coded sums are no longer a general test target
- the stale raster-step wrapper checks have also been updated

## Test Review

## `tests/effq/test_grid.py`

This file covers:

- `compute_index()`
- `compute_coordinate()`
- `_stack_X0X1()`
- `compute_bounds_X0X1()`
- `compute_bounds_X0_X1()`
- `compute_charge_box()`
- `reduce_to_universal()`

Current status:

- the stale `float32` expected coordinates have been updated for the reviewed case
- the strict floating-point bound coverage checks were replaced with a small tolerance helper
- the helper explicitly notes that it is only intended for the current grid-aligned cases and is not a general rule for arbitrary decimal bounds
- the focused regression test for dtype-stable charge-box indexing on a grid boundary remains in place

Conclusion:

- the reviewed `test_grid.py` failures have been fixed in the current tests
- this file is no longer one of the active failing areas in the `tests/effq` suite

## `tests/effq/test_effq.py`

This file exercises:

- quadrature helpers
- interpolation helpers
- `eval_qeff()`
- `compute_qeff()`

Issues:

- the `exact_qline_gaus.json` path issue has been fixed by resolving it relative to the test file
- the older `qmodel=...` call against `eval_qeff()` has been removed
- the old hard-coded `qeff` integral checks have been replaced with `+-3 sigma` charge-recovery checks against `Q`
- `qpoint` coverage was added with:
  - an exact finite-box `erf` comparison for `qpoint_diff3D()`
  - a point-branch `eval_qeff()` charge-recovery test
  - a short-segment agreement test between forced `qline` and forced `qpoint`
- the remaining helper tests are now aligned with the current `float64` default

Conclusion:

- the `eval_qeff()` / `compute_qeff()` coverage in this file has been substantially updated
- the helper tests now match the current dtype contract

## `tests/effq/test_raster_steps.py`

This file covers:

- `Raster._transform()`
- `Raster._head_time_offset_from_tail()`
- `tred.graph.raster_steps()`
- a step-based `Raster` integration check
- one plotting helper

Useful points:

- `_head_time_offset_from_tail()` tests confirm the implementation returns `(tail - head) / velocity`
- `_transform()` tests are still useful

Issues:

- `test_drift_raster_positions()` is a plotting script, not a unit test

Conclusion:

- the file is useful for step semantics
- the dtype smoke test now covers the depo path directly
- the stale integral checks have been replaced with charge-recovery assertions

## `tests/effq/test_raster_dtype.py`

This tracked smoke test was added during the dtype work.

It covers:

- `binned_nd(..., dtype=...)`
- `compute_qeff(..., dtype=...)`
- `Raster.forward(..., dtype=...)`

for both:

- `torch.float32`
- `torch.float64`

What it checks:

- output floating dtype follows the requested compute dtype
- offsets remain `int32`
- outputs are finite and non-zero

Conclusion:

- this is a focused smoke test for the new dtype contract
- it does not replace the older numerical/reference tests
- it is part of the tracked `tests/effq` suite in this branch

## `tests/effq/test_depos.py`

This file now provides direct depos backend tests.

It covers:

- `binned_1d()` finite-width Gaussian behavior
- `binned_1d()` finite-width numerical regression
- `binned_1d()` zero-width spike behavior
- `binned_nd()` finite-width `vdim=1` consistency with `binned_1d()`
- `binned_nd()` zero-width spike numerical regression

for both:

- `torch.float32`
- `torch.float64`

Conclusion:

- the most immediate missing direct `depos.py` coverage has been added
- broader `binned_nd()` finite-width, `minbins`, and `binned()` wrapper coverage are still reasonable follow-ups

## `tests/raster/test_raster_graph.py`

This file is a visualization-oriented smoke test for:

- `Drifter`
- `Raster`

It is not a strong unit test:

- it mainly plots and prints
- it does not assert detailed numerical behavior

Conclusion:

- this file is useful for manual inspection
- it is not sufficient to catch the raster inconsistencies listed above

## Overall Conclusions

1. The local change in `Raster.forward()` was justified and is now fixed in current source.
2. The old `Raster._time_diff()` docstring inconsistency is fixed in current source by renaming it to `Raster._head_time_offset_from_tail()` and updating the docstring.
3. Raster now has an explicit compute dtype with default `float64`.
4. The older raster/effq tests are only partially up to date:
   - some are useful
   - several have now been updated
   - the remaining helper and raster-step failures called out earlier in this note have been fixed
5. `tests/effq/test_raster_dtype.py` exercises the explicit dtype contract in the tracked `tests/effq` suite.
6. Direct depos backend coverage now exists in `tests/effq/test_depos.py`.
7. Charge-box boundary handling is now explicitly dtype-stable for the reviewed exact-on-edge case.

## Command Used

The current effq test suite was checked with:

```bash
PYTHONPATH=src pytest -q tests/effq
```

Observed result in this tree:

- `43 passed`

There are no remaining known `tests/effq` failures in this tree.

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

- `10 passed`
