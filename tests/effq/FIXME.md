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

### ~~Docstring mismatch in `Raster._head_time_offset_from_tail()`~~

Fixed in the current source. The old `Raster._time_diff()` helper has been renamed to `Raster._head_time_offset_from_tail()` and now documents that it returns the offset added to tail time to get head time.

~~`Raster._time_diff()` documents:~~

~~> Always return `(head - tail)/v` if `head is not None`.~~

~~But the implementation returns:

- `(tail - head) / velocity`~~

The tests in `tests/effq/test_raster_steps.py` also expect `(tail - head) / velocity`.

Conclusion:

- ~~the code and tests agree~~
- ~~the old `_time_diff()` docstring is stale~~
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

### ~~Inconsistency in `Raster.forward()` before the local change~~

Fixed in the current source: `sigma` is transformed before dispatching to either the depo or step raster backend.

~~`Raster.forward()` says that:

- `tail`, `head`, and `sigma` are provided in spatial coordinates
- the drift-direction component is transformed into time for rasterization~~

~~That was only fully true for the step path.~~

~~Before the current local edit:

- `tail` was transformed before both depos and steps
- `head` was transformed for steps
- `sigma` was transformed only inside the step branch~~

~~That means the depo path used:

- transformed `tail`
- untransformed `sigma`~~

~~This is inconsistent with both:

- the method docstring
- the step path behavior~~

Conclusion:

- ~~the local change that moves the `sigma` transform into the common path is justified~~
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

## ~~Status of the current local source changes~~

This section is no longer a pending local-change note. The described `Raster.forward()` change is present in the current source.

~~Current local diff in `src/tred/graph.py`:

- moves
  - `sigma = self._transform(sigma, None)`
  - `sigma[:, self._tdim] = sigma[:, self._tdim] / torch.abs(self.velocity)`
- from the step-only path to the common path before checking `head is None`~~

~~Why this is justified:

- `Raster` transforms positions into raster coordinates
- `sigma` represents widths in the same coordinate system
- the drift-axis width must therefore be converted from distance-width to time-width for both:
  - depos
  - steps~~

~~Without this change, the depo path mixes transformed positions with untransformed widths.~~

Conclusion:

- ~~the local `Raster.forward()` change is consistent with the documented raster-space contract~~
- ~~it fixes a real inconsistency even though current tests do not cover the depo path well~~
- fixed in current source and covered by `tests/effq/test_raster_dtype.py`

## Problems in `src/tred/raster/depos.py`

These are source-level issues observed during review.

### ~~Indexing bug in `binned_1d()` and `binned_nd()`~~

Fixed in the current source.

~~Earlier in this review, both:

- `binned_1d()`
- `binned_nd()`~~

~~failed on CPU with:~~

```text
IndexError: tensors used as indices must be long, byte or bool tensors
```

~~The issue came from advanced indexing with `int32` tensors at:

- `src/tred/raster/depos.py:123`
- `src/tred/raster/depos.py:213`~~

This has now been fixed by converting those indexing tensors to `long` at the indexing sites.

Conclusion:

- ~~this bug is no longer considered outstanding~~
- fixed

### ~~Scalar-grid branch in `binned_nd()`~~

Fixed in the current source.

~~`binned_nd()` documents:

- `grid` may be a real scalar or a 1D N-vector~~

~~But in the scalar branch it builds:

- `torch.tensor([grid], dtype=index_dtype, ...)`~~

~~That was inconsistent with the documented meaning of `grid` as a spacing value.~~

This has now been changed to use the raster compute dtype instead of `index_dtype`.

Conclusion:

- ~~this issue is no longer considered outstanding~~
- fixed

## Problems in `src/tred/raster/steps.py`

### `eval_qeff()` test API mismatch

`eval_qeff()` now branches internally using:

- `qline_model`
- `qpoint_model`

But one test still calls it with:

- `qmodel=...`

That currently leads to:

```text
TypeError: tred.raster.steps.eval_qmodel() got multiple values for keyword argument 'qmodel'
```

Conclusion:

- the test no longer matches the implementation interface

Relevant location:

- `tests/effq/test_effq.py`

### ~~Dtype-dependent boundary behavior~~

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

### Hard-coded integral expectations are stale

For current code in this tree:

- `compute_qeff(..., npoints=(2,2,2))` gives about `0.29399486735385094`
- `compute_qeff(..., npoints=(4,4,4))` gives about `0.29597755499800044`

Some tests still expect about:

- `0.3137`
- `0.313723`

Conclusion:

- these reference values do not match the current implementation
- either the implementation changed, or the reference numbers were produced under a different convention/setup

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

Issues:

- several assertions compare `float32` expected tensors against `float64` outputs from `steps.py`
- one strict bound assertion fails on a floating-point roundoff edge

Conclusion:

- this file is partly stale with respect to current dtype behavior
- it now includes a focused regression test for dtype-stable charge-box indexing on a grid boundary

## `tests/effq/test_effq.py`

This file exercises:

- quadrature helpers
- interpolation helpers
- `eval_qeff()`
- `compute_qeff()`

Issues:

- opens `exact_qline_gaus.json` by relative path and fails from repo root
- assumes `float32` in multiple places while implementation uses `float64`
- still uses `qmodel=...` against `eval_qeff()`
- contains stale hard-coded integral expectations

Conclusion:

- multiple tests in this file are outdated or brittle

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

- ~~there is no real unit test of the depo path:~~
  - ~~`Raster.forward(sigma, time, charge, tail, head=None)`~~
- the integral checks use stale reference values
- `test_drift_raster_positions()` is a plotting script, not a unit test

Conclusion:

- the file is useful for step semantics
- ~~it does not protect the depo path affected by the current local change~~
- the depo path is now covered by `tests/effq/test_raster_dtype.py`

## `tests/effq/test_raster_dtype.py`

This file was added during the dtype work.

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

## `tests/effq/test_depos.py`

This file now provides direct depos backend tests.

It covers:

- `binned_1d()` finite-width Gaussian behavior
- `binned_1d()` finite-width numerical regression
- `binned_1d()` zero-width spike behavior
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

1. ~~The local change in `Raster.forward()` is justified.~~ Fixed in current source.
2. ~~The clearest docstring inconsistency is in the old `Raster._time_diff()`.~~ Fixed in current source by renaming it to `Raster._head_time_offset_from_tail()` and updating the docstring.
3. Raster now has an explicit compute dtype with default `float64`.
4. The older raster/effq tests are only partially up to date:
   - some are useful
   - several are stale
   - most still assume legacy dtype and reference-value behavior
5. The new dtype smoke test improves coverage of the explicit dtype contract.
6. Direct depos backend coverage now exists in `tests/effq/test_depos.py`.
7. Charge-box boundary handling is now explicitly dtype-stable for the reviewed exact-on-edge case.

## Command Used

The current raster-related test subset was checked with:

```bash
PYTHONPATH=src pytest -q \
  tests/effq/test_grid.py \
  tests/effq/test_effq.py \
  tests/effq/test_raster_steps.py \
  tests/raster/test_raster_graph.py
```

Observed result in this tree:

- `14 failed, 9 passed`

Those failures are still dominated by stale raster-suite expectations and API drift.

The focused dtype smoke test was checked with:

```bash
PYTHONPATH=src pytest -q tests/effq/test_raster_dtype.py
```

Observed result in this tree:

- `6 passed`

The focused depos backend tests were checked with:

```bash
PYTHONPATH=src pytest -q tests/effq/test_depos.py
```

Observed result in this tree:

- `8 passed`
