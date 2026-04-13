# TRED GPU Runtime Efficiency

> Read-only examination.  No source files were modified.
> Impact tags: **high** / **medium** / **low**.
> Change tags: **quick-win** (few lines) / **structural** (design change).

See [01_algorithms.md](01_algorithms.md) for algorithm context.

---

## Overview: where time goes

The repo root `itpc_timing_summary.{csv,json}` and `pie_chart_*.png` show
prior profiling.  The most expensive stages are typically (in rough order):
1. Convolution (`convo`) — repeated FFT per lace pair per chunk.
2. Raster (`raster`) — float64, per-step computation.
3. ChunkSum charge/current — scattered sparse accumulation.

The findings below address inefficiencies in each stage.

---

## EFF-01 — Float64 in entire step-raster module   {#fp64}

**Impact:** high  
**Change:** structural (precision study needed)  
**Location:** `src/tred/raster/steps.py:11`

```python
float_dtype = torch.float64
```

All tensors allocated in `steps.py` are float64.  On consumer/mid-range
CUDA GPUs (e.g. RTX 3090/4090, A100 40GB), fp64 throughput is
**1/32× to 1/64×** the fp32 throughput.  This single line multiplies the
effective compute time of the entire rasterisation stage by up to 64.

The precision requirement comes from the analytical integrand
`qline_diff3D_script` which involves differences of erfs and ratios of
large/small numbers.  A study to determine whether fp32 + compensated
summation is sufficient, or whether only the critical sub-expressions need
fp64, would allow a mixed-precision strategy.

**Immediate win:**  
Change `float_dtype = torch.float32` and run validation tests to identify
which sub-expressions lose accuracy.  Even partial fp32 (compute in fp32,
accumulate in fp64) would help.

---

## EFF-02 — Response FFT recomputed on every forward call   {#fft-cache}

**Impact:** high  
**Change:** quick-win  
**Location:** `src/tred/convo.py:385`

```python
meas_ = torch.fft.fftn(res_[None,...], dim=dims) * torch.fft.fftn(meas_, dim=dims)
```

Inside `interlaced_symm_v2`, the response lace `res_` is FFT'd on every call
to `forward()`.  The response tensor is constant across all batches and
events; its FFT should be computed once and cached.

The function docstring and existing code have a `# fixme: allow for response
to be pre-FFT'ed` comment (`graph.py:453`), confirming the intent.

The response has shape `(90, 90, 6400)`.  Each lace extracted from it has
shape roughly `(9, 9, 6400)`.  With `nimperpix/2 = 5` pairs per axis and
two spatial axes, there are 25 lace pairs per forward call.  Caching all 25
FFT'd laces (computed once at construction time) would eliminate ~25 FFT
calls per batch.

**Implementation sketch:**  
Add a `_precompute_response_ffts(response, o_shape)` method to `LacedConvo`
that runs `deinterlace_pairs` on the response and stores the FFT'd laces.
Call it in `LacedConvo.__init__` or lazily on first forward call.

---

## EFF-03 — Python for-loop in `accumulate_nd_blocks_v1`   {#v1-loop}

**Impact:** high  
**Change:** quick-win (use `_v2` instead)  
**Location:** `src/tred/sparse.py:308–311`

```python
for u, up in enumerate(unique_pairs):
    mask = (idx_exp == up).all(dim=-1)
    out[u] = Xb[mask].sum(dim=0)
```

`accumulate_nd_blocks_v1` iterates over every unique chunk location in a
Python loop, launching one boolean mask + sum kernel pair per unique chunk.
For a realistic event with thousands of unique chunks, this is thousands of
sequential CUDA kernel launches, each with GPU→CPU sync for the boolean check.
This is the worst possible pattern for GPU utilisation.

`accumulate_nd_blocks_v2` (`sparse.py:315`) uses `scatter_add_` and
`index_add_` — single parallel operations — and should be dramatically faster.

**Recommendation:**  
Default `ChunkSum` should not use `method='chunksum_inplace_v1'` for
production.  Ensure `_v2` or the `chunkify + accumulate` path is used.

---

## EFF-04 — CUDA-CPU sync in `chunkify2` via `.tolist()`   {#chunkify2-sync}

**Impact:** medium  
**Change:** quick-win  
**Location:** `src/tred/sparse.py:264–265`

```python
# FIXME: CUDA-CPU SYNC
data = torch.zeros((tile_keys.size(0),) + tuple(cshape.tolist()), ...)
```

`cshape.tolist()` forces the CUDA device to synchronise with the CPU to
transfer the shape integers.  Since `cshape` is the fixed chunk shape
(set at construction time), it could be cached as a Python tuple once at
init time and passed directly.

**Fix:**
```python
# In __init__: self._chunk_shape_tuple = to_tuple(chunk_shape)
data = torch.zeros((tile_keys.size(0),) + self._chunk_shape_tuple, ...)
```
This is already done in `ChunkSum.__init__` (`graph.py:327`):
```python
self._chunk_shape_tuple = to_tuple(chunk_shape)
```
The cached value just needs to be threaded through to `chunkify2`.

---

## EFF-05 — `steps.cpu()` in `interlaced_symm_v2` hot path   {#convo-cpu-sync}

**Impact:** medium  
**Change:** quick-win  
**Location:** `src/tred/convo.py:352, 359`

```python
c_shape = dft_shape(torch.tensor(signal.data.shape[1:]).to(steps.device)//steps,
                    torch.tensor(response.shape).to(steps.device)//steps)
...
nrm1 = to_tensor(response.shape, device='cpu') // steps.cpu() - 1
```

Two patterns force CPU↔GPU transfers on every call:
- `torch.tensor(signal.data.shape[1:])` creates a CPU tensor from Python ints,
  then `.to(steps.device)` moves it.  `signal.data.shape` is always a Python
  tuple of ints; the division by `steps` can be done in Python arithmetic
  without any tensor.
- `steps.cpu()` copies `steps` from GPU to CPU just to do integer division.
  Since `steps` is a constant (set at `LacedConvo` construction), it should
  be stored as a Python tuple.

**Fix direction:**  
Precompute these shapes at construction time and store as Python tuples or
CPU tensors.

---

## EFF-06 — `torch.cuda.empty_cache()` inside the batch loop   {#empty-cache}

**Impact:** medium  
**Change:** quick-win (remove)  
**Location:** `src/tred/graph.py:382, 428`

```python
torch.cuda.empty_cache()
```

`empty_cache()` releases GPU memory back to the OS allocator.  It is
expensive (causes CUDA context synchronisation and cache flush), and in a
tight batch loop it prevents PyTorch's memory allocator from reusing
recently-freed blocks.  It should only be called when genuinely OOM and
needing to free memory for a new allocation.

The surrounding code in `_chunksum` / `_chunksum2` is already behind a
`nchunks > 1` guard, but `nchunks == 1` for most events, and the code after
`return accumulate(chunkify(block, self.chunk_shape))` on `graph.py:360` is
**unreachable dead code** (the `return` statement on that line exits the
function before the loop is entered).

**Immediate action:**  
The entire body of `_chunksum` from line 362 onward is dead code.  Remove it.
The `empty_cache` calls go with it.

---

## EFF-07 — `torch.zeros(...).to(dat.device)` allocates then copies   {#zeros-device}

**Impact:** medium  
**Change:** quick-win  
**Location:** `src/tred/chunking.py:150`

```python
summed = torch.zeros((unique_locs.shape[0], *dat.shape[1:]), dtype=dat.dtype).to(dat.device)
```

`torch.zeros(...)` without a `device=` argument allocates on CPU, then
`.to(dat.device)` copies to GPU.  For large chunks this creates a
CPU-side allocation and a transfer that can be avoided:

```python
summed = torch.zeros((unique_locs.shape[0], *dat.shape[1:]),
                     dtype=dat.dtype, device=dat.device)
```

Same pattern likely occurs elsewhere when `torch.zeros` / `torch.ones` are
called without explicit `device=`.

---

## EFF-08 — Per-dimension Python loop in `raster/depos.py`   {#depos-loop}

**Impact:** medium  
**Change:** structural  
**Location:** `src/tred/raster/depos.py:193–214`

```python
for dim in range(vdims):
    ...
    rel_grid_ind = torch.linspace(0, 2*dim_n_half, 2*dim_n_half+1).to(device=device)
    ...
```

Two inefficiencies in one loop:
1. `torch.linspace(...)` returns a CPU tensor; `.to(device=device)` copies it
   to GPU inside the loop.  Use `device=device` in the `linspace` call.
2. The loop is a Python-level serialisation over spatial dimensions (vdim=3).
   While only 3 iterations, each iteration is a separate set of CUDA kernels
   launched sequentially.  This is a small but avoidable overhead.

The comment in the code acknowledges it: `"Suffer per-dimension serialization.
We do it because linspace() is 1D only."`

**Fix:**  
Use `torch.meshgrid` to compute all dimension indices simultaneously, then
broadcast the erf computation across all dimensions at once.

---

## EFF-09 — Repeated `contiguous()` allocations in response loading   {#contiguous}

**Impact:** low  
**Change:** quick-win  
**Location:** `src/tred/response.py:142, 146`

```python
full_response = quadrant_copy(raw).contiguous()
...
return response.contiguous()
```

Two `.contiguous()` calls create two full copies of the response tensor.
Only the final `contiguous()` is needed (if the view/reshape chain left a
non-contiguous layout).  The intermediate one on `full_response` is consumed
only by `view`, which requires contiguity, so it is necessary — but the chain
could potentially be restructured to require only one copy.

---

## EFF-10 — Readout while-loop forces repeated GPU-CPU syncs   {#readout-loop}

**Impact:** medium (event-rate dependent)  
**Change:** structural  
**Location:** `src/tred/readout.py:67, 102`

```python
while True:
    ...
    if not triggered.any():
        break
```

`triggered.any()` is a GPU reduction that returns a Python bool, forcing a
GPU→CPU sync on every iteration.  For a typical ND-LAr event with O(100–1000)
pixel firings spread across the time window, the number of iterations is
bounded by the maximum number of ADC triggers per pixel.  Even so, 10–50
iterations × GPU sync per batch is a non-negligible overhead.

**Structural alternative:**  
Compute all possible crossings in one pass by finding all threshold-crossing
times simultaneously with `torch.argmax` and working out the dead-time logic
analytically, eliminating the while-loop.

---

## EFF-11 — `locs.detach().clone()` on every drift call   {#drift-clone}

**Impact:** low  
**Change:** quick-win  
**Location:** `src/tred/drift.py:154`

```python
locs = locs.detach().clone()
```

This clones the full position tensor before modifying `locs[:,vaxis]`.
Since `drift()` is called from `Drifter.forward()` which has already cloned
`tail` in `_ensure_tail_closer_to_anode`, the clone here may be redundant.
If the caller can guarantee no aliasing, `locs[:, vaxis] = target + ...`
can be done on a view instead.

---

## EFF-12 — No `torch.compile` or CUDA graphs   {#compile}

**Impact:** high (potential)  
**Change:** structural (experimental)  

None of the hot path modules use `torch.compile` (inductor backend).
For the convolution and scatter-accumulation stages, `torch.compile` with
the inductor backend may provide 2–5× speedup by fusing elementwise ops and
eliminating intermediate tensor allocations.

A first experiment would be wrapping `LacedConvo.forward` and
`chunking.accumulate` with `torch.compile`.

---

## EFF-13 — No use of `torch.fft.rfft` (real FFT)   {#rfft}

**Impact:** medium  
**Change:** quick-win  
**Location:** `src/tred/convo.py:385, 390`

```python
meas_ = torch.fft.fftn(res_[None,...], dim=dims) * torch.fft.fftn(meas_, dim=dims)
...
meas = torch.fft.ifftn(meas, dim=dims)
```

The signal and response are real-valued tensors (or at least have a
real-valued interpretation before the complex packing trick).  Using
`torch.fft.rfftn` / `torch.fft.irfftn` would halve the size of the FFT and
roughly halve the FFT compute time (the last dimension of the spectrum is
`N//2 + 1` instead of `N`).

The complex packing trick (`sig_lace_complex = torch.complex(...)`) is used
to encode two real signals in one complex FFT; care would be needed to adapt
this to `rfft`, but the principle carries over.

---

## EFF-14 — `hdf_keys` in `loaders.py:58` uses undefined variable   {#hdf-typo}

**Impact:** low (runtime error on bad code path)  
**Location:** `src/tred/loaders.py:62`

```python
obj = h5py.File(fobj)  # 'fobj' is not defined; should be 'obj'
```

This branch (`if isinstance(obj, str)`) would raise a `NameError` if the
function is called with a string path.  The function is likely called with an
already-open `h5py.Group` in practice, so this path is never exercised.

---

## Summary table

| ID | Stage | Impact | Change effort |
|----|-------|--------|---------------|
| EFF-01 | Raster (float64) | **high** | structural |
| EFF-02 | Convo (FFT cache) | **high** | quick-win |
| EFF-03 | Sparse v1 loop | **high** | quick-win |
| EFF-04 | chunkify2 sync | medium | quick-win |
| EFF-05 | convo cpu sync | medium | quick-win |
| EFF-06 | empty_cache in loop | medium | quick-win |
| EFF-07 | zeros device | medium | quick-win |
| EFF-08 | depos dim loop | medium | structural |
| EFF-09 | contiguous alloc | low | quick-win |
| EFF-10 | readout loop sync | medium | structural |
| EFF-11 | drift clone | low | quick-win |
| EFF-12 | no torch.compile | high (potential) | structural |
| EFF-13 | fftn → rfftn | medium | quick-win |
| EFF-14 | hdf_keys typo | low | quick-win |
