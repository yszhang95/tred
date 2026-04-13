# TRED Memory Usage Analysis

> Read-only examination.  No source files were modified.
> See also [01_algorithms.md](01_algorithms.md) for tensor shape context.

---

## 1. Peak Memory Map

The GPU memory high-water mark is reached during the **rasterisation** or
**convolution** stage (confirmed by the `op_w_max_mem` tracking in
`plots/graph_effq.py:70–77`).  The breakdown per stage is:

### 1.1 Raster (steps) — charge boxes

Shape per batch of `N_steps` steps with universal box size `(Sx, Sy, St)`:

```
charge_boxes: (N_steps, Sx, Sy, St)  dtype=float64
```

Typical values: `N_steps = 32768`, `Sx = Sy ≈ 5–30`, `St ≈ 5–50`.

**Example:** 32768 steps × 20 × 20 × 20 voxels × 8 bytes (float64) = **2.1 GB**

This is before any accumulation.  The charge boxes must coexist with the
offset tensor `(N_steps, 3)` (int32, negligible) and intermediate tensors
in `compute_qeff`.

### 1.2 Envelope (chunkify)

`chunkify` allocates a dense `envelope` tensor spanning the bounding box of
all charge boxes:

```
envelope: (Ex, Ey, Et)  dtype=float64
```

where `Ex` = max(location[:,0]) − min(location[:,0]) + Sx (rounded up to
chunk boundary), similarly for other dims.

For a typical event with steps spread over 100 pixels × 100 pixels × 7200
time ticks and a chunk shape of (40, 40, 32), the envelope could be
`(160, 160, 7232)`.

**Example:** 160 × 160 × 7232 × 8 bytes = **1.5 GB**

The charge boxes AND the envelope coexist during `fill_envelope`.

### 1.3 Response tensor

```
response: (90, 90, 6400)  dtype=float32
```

Size: 90 × 90 × 6400 × 4 bytes = **207 MB** (constant, lives for full run).

### 1.4 Convolution intermediates

`interlaced_symm_v2` allocates:
- `sig_lace_complex`: (N_chunks, c_shape[0], c_shape[1], c_shape[2])
  complex64 = 2 × float32.
  With `c_shape ≈ (8, 8, 512×4) = (8, 8, 2048)`, and 50 chunks:
  50 × 8 × 8 × 2048 × 8 bytes ≈ **52 MB** per lace pair.
- `meas`: same shape — accumulator across lace pairs.
- Both alive simultaneously during the lace loop → 2 × 52 MB = **104 MB**
  per batch of 50 chunks.

For larger chunk batches or larger `o_shape`, this scales linearly.

---

## 2. Memory Savings Opportunities

### MEM-01 — Universal box shape inflates charge-box tensor   {#universal-box}

**Savings potential:** high  
**Location:** `src/tred/raster/depos.py:176`,
`src/tred/raster/steps.py:112–124` (`reduce_to_universal`)

Both raster paths compute the largest box needed by any step in the batch
and pad **all** steps to that size.  For a batch containing a mix of MIP
tracks (small sigma) and heavily-diffused electrons (large sigma), the
worst-case box size dominates, wasting memory for most steps.

**Example:**  
If 99% of steps need a 5×5×5 box but 1% need a 25×25×25 box, all steps are
allocated as 25×25×25.  Memory usage is 25× higher than it needs to be for
the majority of steps.

**Mitigation strategies:**

a. **Bucket by box size**: sort steps by sigma, group into buckets with
   similar box sizes, run each bucket separately with its own universal shape.
   Trade: more Python-level loops per batch vs. memory saving.

b. **Per-step ragged boxes with padded batching**: pad to the median + N sigma
   (not the maximum) and clip unusually large steps.

c. **Reduce `N_steps` per batch**: reducing `batch_scheme[0]` from 32768
   reduces the universe of sigmas per batch, reducing the worst-case size.
   Current default is 32768; for steps with very heterogeneous sigma,
   smaller batches (e.g. 4096) may give lower peak memory with only modest
   throughput loss.

---

### MEM-02 — float64 charge boxes double GPU memory   {#fp64-memory}

**Savings potential:** high  
**Location:** `src/tred/raster/steps.py:11`

`float_dtype = torch.float64` means every charge box element uses 8 bytes
instead of 4 bytes for float32.  For the example in §1.1, this is the
difference between 2.1 GB and 1.05 GB for the charge box tensor, and
between 1.5 GB and 0.75 GB for the envelope.

Converting to float32 (or mixed precision) would approximately **halve** the
memory of the rasterisation stage.  See also
[03_gpu_efficiency.md §EFF-01](03_gpu_efficiency.md#fp64).

---

### MEM-03 — Envelope allocation in `chunkify`   {#envelope-alloc}

**Savings potential:** medium  
**Location:** `src/tred/sparse.py:160–167`

`fill_envelope` allocates a dense tensor spanning the full bounding box of
all charge boxes (`sparse.py:119`).  For events with many spatially-spread
charge boxes, this envelope is much larger than the total data in the boxes.

`chunkify2` (`sparse.py:207`) avoids the envelope entirely, using index
arithmetic instead.  The trade-off is a CUDA-CPU sync (see EFF-04) and more
complex index logic.

An alternative is to process `chunkify` on sub-regions of the detector
(streaming over spatial tiles), capping the envelope size.

---

### MEM-04 — `locs.detach().clone()` in `drift.py`   {#drift-clone}

**Savings potential:** low  
**Location:** `src/tred/drift.py:154`

Clones the full `(N_steps, 3)` position tensor on every drift call.
For 32768 steps, this is 32768 × 3 × 4 bytes = 384 KB — small but
unnecessary if the caller guarantees no aliasing.

Similarly `_ensure_tail_closer_to_anode` (`graph.py:188`) clones both
`tail` and `head` before any check of whether a swap is needed:

```python
tail_new, head_new = tail.clone().detach(), head.clone().detach()
```

For 32768 steps × 3 dims × 4 bytes × 2 tensors = 768 KB wasted per batch
when no swap is needed.

---

### MEM-05 — Convolution accumulator keeps complex tensor alive   {#convo-complex-alive}

**Savings potential:** medium  
**Location:** `src/tred/convo.py:383–392`

Inside the lace loop:
```python
res_  = zero_pad(res_lace[0], o_shape)      # real, padded response lace
meas_ = zero_pad(sig_lace_complex, o_shape) # complex, padded signal lace
meas_ = torch.fft.fftn(res_[None,...], ...) * torch.fft.fftn(meas_, ...)
if meas is None:
    meas = meas_
else:
    meas += meas_                            # accumulates into meas
```

`meas_` (shape ~ (N_chunks, c1, c2, c3) complex64) is created fresh each
iteration and added to `meas` (same shape complex64).  Both coexist during
the `+=` operation.  After the loop, `meas` holds the full complex spectrum.

After `ifftn`, only `meas.real` is needed.  But `meas.imag` (the flipped
symmetric partner) persists until the function returns.  The final computation:

```python
meas.real = meas.real + torch.flip(meas.imag, dims=flipdims)
return Block(data=meas.real, ...)
```

...creates a temporary for `torch.flip(meas.imag)` alongside the full complex
`meas`.  Peak usage here is ~3× the real-valued output tensor.

**Fix:** Materialise the result immediately and discard `meas`:
```python
result = meas.real + torch.flip(meas.imag, dims=flipdims)
del meas
return Block(data=result, location=super_location)
```

---

### MEM-06 — Response tensor FFT not cached; recomputed each call   {#response-fft-mem}

**Savings potential:** low (but related to EFF-02)  
**Location:** `src/tred/convo.py:385`

Pre-computing the response FFT (`rfft` preferred) and storing it would add
~207 MB of GPU memory (for float32 response) but eliminate repeated FFT
compute and temporary allocations.  If `rfft` is used, only 106 MB is needed.

The trade-off is: cache 106 MB permanently vs. allocate/free ~52 MB × 25
times per batch.  For a GPU with ≥ 24 GB, the persistent cache is clearly
preferable.

---

### MEM-07 — `BlockSparseBlock.size()` triggers CPU sync   {#block-size-sync}

**Savings potential:** low (correctness concern)  
**Location:** `src/tred/blocking.py:52–56`

```python
def size(self):
    '''
    Return torch.Size like a tensor.size() does. This includes batched dimension.

    Warning: potential CPU-CUDA synchronization per call.
    '''
    return Size([self.nbatches] + self.shape.tolist())
```

`self.shape.tolist()` on a GPU tensor forces a CPU sync.  `Block.shape` is
an integer tensor set at construction time from `data.shape[1:]` — which is
always a Python int tuple.  The tensor could be kept as a Python tuple
throughout, avoiding the sync entirely.

---

## 3. Peak Memory Estimates for a Typical ND-LAr Batch

Assuming: N_steps=32768, universal box 20×20×30, chunk_shape=(40,40,32),
response (90,90,6400), N_chunks≈2000 after accumulation:

| Tensor | Size | Dtype | GPU MB |
|--------|------|-------|--------|
| Charge boxes | 32768 × 20×20×30 | float64 | **1180** |
| Charge-box offsets | 32768 × 3 | int32 | 0.4 |
| Envelope (chunkify) | ~160×160×7232 | float64 | **1326** |
| Signal Block (after ChunkSum) | 2000 × 40×40×32 | float32 | **390** |
| Response tensor | 90×90×6400 | float32 | **207** |
| Convo meas (complex) | 2000×8×8×2048 | complex64 | **2048** |
| **Estimated peak** | | | **~5000 MB** |

This comfortably exceeds 4 GB and approaches 8 GB.  The 24 GB target
(from `README.org`) is met, but with significant headroom consumed by the
float64 and the envelope.  Converting to float32 throughout and optimising
the envelope (MEM-01, MEM-02, MEM-03) could bring peak below 3 GB.

---

## 4. Quick Wins for Memory Reduction

Ordered by expected saving with minimal risk:

1. **float32 in `raster/steps.py`** — halves raster stage memory (~1.2 GB).
   Risk: precision — needs validation.
2. **`del meas` after ifftn in `convo.py`** — frees the accumulator complex
   tensor (~2 GB at peak) before returning real part.  No correctness risk.
3. **Reduce batch size** (`batch_scheme[0]`) — immediate, configuration-only.
   Halving from 32768 to 16384 steps reduces charge-box tensor by ~2× and
   reduces the worst-case envelope.
4. **Fix `torch.zeros(...).to(device)`** (`chunking.py:150`) — avoids a CPU
   buffer allocation for the accumulate tensor.
5. **Cache response FFT** (`convo.py`) — saves ~52 MB × 25 re-allocations per
   batch at the cost of 106 MB persistent.
