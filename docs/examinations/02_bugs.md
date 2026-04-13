# TRED Potential Bugs

> Read-only examination.  No source files were modified.
> Severity tags: **likely** (probably wrong), **possible** (context-dependent),
> **nit** (not a bug, but an easy-to-hit trap).
> Each entry gives: location · summary · why it looks wrong · how to confirm · suggested fix direction.

---

## BUG-01 — `fill_envelope` overwrites instead of accumulating   {#fill-envelope}

**Severity:** likely  
**Location:** `src/tred/sparse.py:131`

```python
envelope.data.flatten()[inds] = block.data.flatten()
```

**Problem:**  
This is an assignment (`=`), not an accumulation (`+=`).  If two input blocks
map to the same voxel inside the envelope (i.e., `inds` contains duplicates),
the second value silently clobbers the first with no error or warning.

**Why this might be acceptable:**  
`fill_envelope` is called from `chunkify` (`sparse.py:167`) with a single
`block` argument at a time, not in a loop.  `chunkify` first builds the
envelope from the full block's bounding box, then fills it once.  The danger
arises if the *source block itself* has two batch elements mapping to the same
envelope voxel, which can happen when charge boxes overlap (they often do for
nearby steps).

**How to confirm:**  
Check whether `inds` returned by `indexing.crop_batched` can ever contain
duplicate entries for a realistic set of overlapping charge boxes.  A quick
diagnostic: compute `len(inds) > len(torch.unique(inds))` before the
assignment.

**Fix direction:**  
Replace the assignment with:
```python
envelope.data.flatten().scatter_add_(0, inds, block.data.flatten())
```
or use `index_add_`.

---

## BUG-02 — `meas.real = …` assignment on a complex tensor   {#convo-real-assign}

**Severity:** possible (version-dependent)  
**Location:** `src/tred/convo.py:391`

```python
meas.real = meas.real + torch.flip(meas.imag, dims=flipdims)
```

**Problem:**  
In PyTorch, `Tensor.real` on a complex tensor returns a *view* into the real
part, and assigning to it via `=` rebinds the Python name `meas.real` to a
new tensor rather than modifying the complex tensor in-place.  As a result
the intended "add the flipped imaginary part into the real part" may silently
be a no-op (or raise depending on PyTorch version).

The function then returns `Block(data=meas.real, ...)`, which would return
only the real part — so if the assignment is a no-op the imaginary
contribution (encoding the "flipped symmetric lace") is lost, giving wrong
induced-current values.

**How to confirm:**  
Run a 1-D or 2-D toy simulation with known symmetry, check the returned
currents are non-zero on the symmetric side.  Alternatively inspect
`meas.real` before and after line 391 to see whether it changes.

**Fix direction:**  
Use `torch.view_as_real`, operate on the view, then `torch.view_as_complex`,
or simply:
```python
result = meas.real + torch.flip(meas.imag, dims=flipdims)
return Block(data=result, location=super_location)
```

---

## BUG-03 — Absolute threshold `1e-3` in `accumulate` drops valid charge   {#accumulate-threshold}

**Severity:** possible  
**Location:** `src/tred/chunking.py:157`

```python
indices = torch.where((torch.abs(summed) > 1e-3).any(dim=[i+1 for i in range(chunk.vdim)]))
```

**Problem:**  
The code drops any chunk whose maximum absolute value is ≤ 1e-3.  This
threshold is dimensionally meaningless: charge values are in units of
electrons (typical ~ 10–10,000 e⁻), but induced current values after
convolution can be in different units and can legitimately be much smaller
than 1e-3 (e.g. if the response is in units of e⁻/tick and currents are
fractional electrons per 50 ns tick, typical values are ~0.001–0.1).

In the readout stage, chunks with very small currents can be real signal if
the pixel is near threshold.  Dropping them will mis-model near-threshold
behaviour.

**How to confirm:**  
Add a counter before and after the `indices` filter that prints the number
of dropped chunks.  For a high-occupancy event, non-zero drops could indicate
suppression of small-signal voxels.

**Fix direction:**  
Use a relative threshold (max|summed| / max|dat| > epsilon) or remove the
filter entirely and let the readout threshold do the zero-suppression.

---

## BUG-04 — Integer truncation in `absorb` causes biased electron absorption   {#absorb-int}

**Severity:** likely  
**Location:** `src/tred/drift.py:96–102`

```python
charge = charge.to(dtype=torch.int32)      # line 96
...
return torch.where(dt>=0, charge * torch.exp(-dt / lifetime), charge)  # line 102
```

**Problem:**  
`charge * torch.exp(-dt/lifetime)` computes a float, but `charge` is int32
and PyTorch performs `where` with the dtype of the first truthy branch.
For small drift times, `exp(-dt/lifetime) ≈ 0.9999...` and the result cast
to int32 is truncated to the original integer (or in some PyTorch versions
the multiply is promoted to float then `where` returns float — behaviour
differs by version).

More critically, when `fluctuate=False` (the default), the absorption
`charge * exp(-dt/lifetime)` is the **mean** expected charge, not an integer.
Truncating to int32 introduces a systematic downward bias of up to 1 electron
per step, which accumulates across many steps.

**How to confirm:**  
Pass `charge = torch.tensor([1000], dtype=torch.int32)` with
`dt = torch.tensor([0.01])` and `lifetime = 1.0`.
Check whether the returned value is 990 (correct round) or 990 (truncation).
Also inspect the dtype of the return value.

**Fix direction:**  
Keep the computation in float32 until after any rounding:
```python
survived = (charge.to(torch.float32) * torch.exp(-dt / lifetime))
return torch.where(dt >= 0, survived.round().to(torch.int32), charge)
```

---

## BUG-05 — `_ensure_tail_closer_to_anode` uses fragile in-place swap   {#tail-swap}

**Severity:** nit (currently correct but fragile)  
**Location:** `src/tred/graph.py:205`

```python
tail_new[swap_idx], head_new[swap_idx] = head_new[swap_idx].clone(), tail_new[swap_idx].clone()
```

**Problem:**  
Python evaluates the right-hand side as a tuple before assigning; the
`.clone()` calls prevent aliasing.  The code works correctly today.  However:
1. The implementation clones *both* `tail` and `head` unconditionally at
   `graph.py:188`, even when no swap is needed (e.g. `swap_idx` is empty).
   This doubles the memory used for positions on every batch.
2. The pattern is non-obvious and easy to break: if a future editor removes
   either `.clone()`, the swap silently becomes wrong (both sides alias the
   same modified tensor).

**Fix direction:**  
Use a temporary buffer:
```python
tmp = tail_new[swap_idx].clone()
tail_new[swap_idx] = head_new[swap_idx]
head_new[swap_idx] = tmp
```
Or avoid the unconditional clone by checking whether any swap is needed
before allocating `tail_new` / `head_new`.

---

## BUG-06 — NaN silencing in diffusion masks real upstream errors   {#nan-mask}

**Severity:** possible  
**Location:** `src/tred/drift.py:77`

```python
sigma[torch.isnan(sigma)] = 0
```

**Problem:**  
NaNs in `sigma` arise when `dt < 0` (a step drifts "backward") or when
`sigma=None` and `dt=0` (zero-time diffusion).  Setting them silently to 0
hides the underlying issue and causes downstream effects:
- Zero sigma is handled with a special `spikes` path in `raster/depos.py:113`
  (point-charge grid assignment) — meaning a step that should have some
  spatial extent is treated as a point charge.
- Steps with negative drift time (possibly from bad geometry or floating-point
  rounding near the anode) are treated as zero-width, creating spurious
  point-like depositions.

**Fix direction:**  
At minimum, log a warning when NaNs are found.  Better: clamp `dt` to ≥ 0
before computing sigma, and separately track which steps have `dt < 0`
(they should be dropped as unphysical).

---

## BUG-07 — `concatenate_waveforms` uses `index_put_` with `accumulate=False`   {#waveform-overwrite}

**Severity:** possible  
**Location:** `src/tred/plots/graph_effq.py:136`

```python
wf_out.index_put_((flat_batch, flat_time), flat_values, accumulate=False)
```

**Problem:**  
If two current-block entries map to the same `(pixel, time_tick)` the second
overwrites the first (no accumulation).  This should not happen if the sparse
current blocks are properly non-overlapping by this point, but it is not
guaranteed by any assertion.  Silent data corruption if the assumption is
violated.

**Fix direction:**  
Either use `accumulate=True` (correct in all cases) or add an assertion that
no `(flat_batch, flat_time)` pair is repeated.

---

## BUG-08 — `nd_readout` return path for empty hits is unreachable   {#readout-dead-code}

**Severity:** nit  
**Location:** `src/tred/readout.py:141`

```python
if len(olocs) == 0:
    return torch.zeros((0, len(pixel_axes)+3), ...), torch.zeros((0,), ...)
    raise NotImplementedError("Not sure how to handle empty hit collection")
```

**Problem:**  
The `raise NotImplementedError` is dead code: the `return` statement on the
preceding line means it is never reached.  The comment suggests the developer
left it as a placeholder, but it gives a false impression that empty
collections are unhandled.

**Fix direction:**  
Remove the dead `raise` line; the existing `return` is correct.

---

## BUG-09 — `readout.py` `start` initialisation ignores leftover charge   {#readout-leftover}

**Severity:** possible  
**Location:** `src/tred/readout.py:48–49`

```python
# FIXME: start should be initialized according to leftover
start = torch.zeros(...)
```

**Problem:**  
When events are processed in batches, charge that accumulates up to the end
of the batch window but doesn't cross threshold ("leftover") is discarded.
The `leftover` parameter exists in the function signature but raises
`NotImplementedError` if not None.  This means near-threshold signals split
across batch boundaries can be missed, which affects trigger efficiency.

**Fix direction:**  
Implement the leftover accumulator.  This requires tracking `Xacc[:,-1]` at
the end of each batch and passing it as the initial value for the next.

---

## BUG-10 — `chunkify2` type-check on `stride` dtype is self-referential   {#chunkify2-typeguard}

**Severity:** nit  
**Location:** `src/tred/sparse.py:174–175`

```python
if stride.dtype != stride.dtype != offset_dtype:
```

**Problem:**  
The condition is `stride.dtype != stride.dtype` which is always `False`
(a value is always equal to itself), making the entire check a no-op.
The intended check was probably `stride.dtype != offset_dtype`.

**Fix direction:**
```python
if stride.dtype != offset_dtype:
```
