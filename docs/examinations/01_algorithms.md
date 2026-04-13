# TRED Algorithm Explanations

> Read-only examination. No source files were modified.
> See also [00_overview.md](00_overview.md) for data-flow context.

---

## 1. Recombination (`recombination.py`)

### What it does
Converts energy deposition (dE in MeV) and stopping power (dE/dx in MeV/cm)
into an expected number of ionisation electrons surviving recombination.

Two models are implemented:

**Birks model** (`recombination.py:13`):
```
R = A / (1 + dEdx * k / (E * rho))
Q = R * dE / Wi
```

**Modified Box model** (`recombination.py:37`):
```
R = ln(A + B*dEdx/(E*rho)) / (B*dEdx/(E*rho))
Q = R * dE / Wi
```

Both are fully vectorised over (N_steps,). No Python loops. No GPU sync.
Typical output dtype: float32.

### Key tensors
| Tensor | Shape | Dtype |
|--------|-------|-------|
| dE, dEdx | (N_steps,) | float32 |
| charge Q out | (N_steps,) | float32 |

### Cost
Low: two element-wise operations over N_steps. Memory: O(N_steps).

---

## 2. Drift (`drift.py`)

### What it does
Models three physical processes for each step / depo:
1. **Transport** (`drift.py:20`): computes drift time `dt = (target − loc) / v`
   along the drift axis (default x, `vaxis=0`).
2. **Diffusion** (`drift.py:32`): computes post-drift Gaussian sigma via
   `sigma = sqrt(2*D*dt + sigma0^2)` (quadrature addition of longitudinal and
   transverse diffusion).  Separate diffusion coefficients are supported per
   dimension.
3. **Absorption** (`drift.py:82`): applies electron lifetime attenuation
   `Q_out = Q_in * exp(-dt / lifetime)`.  With `fluctuate=True` applies a
   Binomial fluctuation on top.

All three operations are vectorised over (N_steps,).  The `Drifter` nn.Module
(`graph.py:67`) wraps these and adds a swap to ensure the "tail" is always
the point closer to the anode.

Optional time-shift correction `drtoa` converts anode-plane drift time to
response-plane drift time.

### Key tensors
| Tensor | Shape | Dtype |
|--------|-------|-------|
| locs in/out | (N_steps,) or (N_steps, vdim) | float32 |
| times in/out | (N_steps,) | float32 |
| sigma out | (N_steps, vdim) | float32 |
| charge out | (N_steps,) | int32 → float32 |

### Cost
Low: all element-wise or broadcast operations. Memory: O(N_steps × vdim).
Notable: `locs.detach().clone()` copies the position tensor every call
(`drift.py:154`). See [04_memory.md](04_memory.md#drift-clone).

---

## 3. Raster — Depos (`raster/depos.py`)

### What it does
For point-like depos, computes the effective charge on each grid voxel by
integrating the 3-D Gaussian distribution over each voxel.

The N-D version (`binned_nd`, `raster/depos.py:128`) works as follows:

1. Compute per-depo, per-dimension half-span `n_half`:
   how many grid points are needed to cover ±nsigma×sigma.  **Critically,
   the maximum over all depos is taken**, creating a universal box shape.
2. Build the `grid0` lower corner for each depo.
3. Loop over spatial dimensions (Python `for` loop, `raster/depos.py:193`).
   For each dimension:
   a. Construct relative grid point indices via `torch.linspace`.
   b. Convert to absolute coordinates.
   c. Compute `erf` values at bin edges.
   d. Compute bin integrals as half the difference of adjacent erfs.
4. Form the N-D charge box as the outer product of per-dimension integrals via
   `torch.einsum` (for N=2 or N=3).
5. Multiply by Q.

**Output:** `(qeff, grid0)` where `qeff` has shape `(N_depos, 2*n_half[0]+1,
2*n_half[1]+1, 2*n_half[2]+1)` and `grid0` has shape `(N_depos, vdim)`.

### Key property: universal (worst-case) box shape
Because all depos are padded to the same box shape (maximum sigma depo),
depos with small diffusion waste most of their allocated voxels.  This is
a significant memory amplification when sigma varies widely. See
[04_memory.md](04_memory.md#universal-box).

---

## 4. Raster — Steps (`raster/steps.py`)

### What it does
For line-segment steps, computes the charge distribution by integrating the
**anisotropic 3-D Gaussian smeared along a line segment** over each grid
voxel.  This is substantially more complex than point depos.

The key function is `compute_qeff` which calls `compute_charge_box` then
one of two evaluation methods:

**Method `gauss_legendre`** (default):
Uses Gauss–Legendre quadrature to numerically integrate the charge density
`qline_diff3D` (`steps.py:190`) at GL nodes within each grid voxel, then
sums the weighted values.  The analytical integrand is:

```
q(x,y,z) = Q / (4π Δ) * exp(-sy²(x*dz01 + ...) / (2Δ²))
          * exp(...) * exp(...) * [erf(·) - erf(·)]  / (sqrt(2) Δ sx sy sz)
```
where `Δ = sqrt(sy²sz²*dx01² + sx²sy²*dz01² + sx²sz²*dy01²)` (the
denominator encoding the line-spread geometry).  This is computed as a
TorchScript function (`qline_diff3D_script`, `steps.py:216`) for JIT speed.

The GL quadrature uses `npoints=(2,2,2)` by default (8 evaluation nodes per
voxel), with weights pre-computed once via `create_wu_block`.

**Short step fallback**: steps shorter than 5% of the sigma along any
dimension are treated as point depos (`qpoint_diff3D`, `steps.py:279`).

### Universal shape
Same issue as depos: `reduce_to_universal` (`steps.py:112`) collapses all
per-step box shapes to the per-batch maximum before allocating, so every
step's charge box is padded to the worst-case size.
See [04_memory.md](04_memory.md#universal-box).

### Float64 usage
The entire `steps.py` module uses `float_dtype = torch.float64` (line 11).
The analytical integrand requires precision, but fp64 on consumer CUDA GPUs
runs at 1/32× of fp32 throughput.
See [03_gpu_efficiency.md](03_gpu_efficiency.md#fp64).

---

## 5. Block Data Structure (`blocking.py`)

A `Block` is the central data container throughout TRED.  It holds:
- `location`: (N_batches, vdim) int32 tensor — absolute grid-index of the
  lower corner of each volume.
- `data`: (N_batches, d1, d2, ..., dN) float tensor — values on the volume.

All volumes in one Block share the same `shape` (d1,d2,...,dN), so a Block
is essentially a dense batched tensor of rectangular volumes at arbitrary
sparse locations.  This is the "Block Sparse Binned" representation described
in `docs/concepts.org`.

Key methods: `size()` (with CPU sync warning), `vdim`, `nbatches`.
Helper: `concat_blocks(blocks)` — concatenates along the batch dimension.

---

## 6. Block-Sparse Accumulation (`sparse.py` / `chunking.py`)

### The central problem
After rastering, each step produces one charge box (a Block batch element)
at an arbitrary grid location.  Many boxes overlap on the grid.  The goal
is to **merge overlapping boxes** onto a coarser "chunk" grid, summing
charges at common voxels.

### chunkify (sparse.py:160)
1. Create a `SGrid` with spacing = `chunk_shape`.
2. Compute the smallest aligned "envelope" Block containing all input blocks.
3. Call `fill_envelope`: expand the (possibly empty) envelope tensor and
   write each input block into it at the correct offset using flat indexing
   (`crop_batched`).
4. Call `reshape_envelope`: view the envelope as a grid of chunks, return
   a new Block where each batch element is one chunk.

**Key limitation**: `fill_envelope` **assigns** (not accumulates) values into
the envelope (`sparse.py:131`).  This means if two input blocks map to the
same voxel in the envelope, the second silently overwrites the first.
Whether this is safe depends on the invariant that `chunkify` is called with
non-overlapping source blocks — see [02_bugs.md](02_bugs.md#fill-envelope).

### accumulate (chunking.py:140)
After chunkify produces a Block of chunks (some at identical locations):
1. `torch.unique(loc, dim=0)` finds distinct chunk locations.
2. `index_add_(0, inverse, data)` accumulates data from duplicate chunks
   into a single output.
3. Non-zero chunks (|val| > 1e-3) are retained.

`index_add_` is a single parallel GPU operation — efficient.

### Alternative: chunkify2 (sparse.py:207)
A second, scatter-based implementation that does not build an explicit
envelope.  Instead it computes flat tile indices for every element of every
block and uses a two-step `scatter_add_ + index_add_` to accumulate directly.
This avoids the large envelope allocation but has a CUDA-CPU sync hazard
(`sparse.py:265`).  See [03_gpu_efficiency.md](03_gpu_efficiency.md#chunkify2-sync).

### accumulate_nd_blocks_v1 / v2 (sparse.py:275 / 315)
Two in-place variants that skip the envelope entirely and operate on
the raw Block data tensor.  `_v1` has a Python for-loop over unique chunk
indices (very slow; see [03_gpu_efficiency.md](03_gpu_efficiency.md#v1-loop)).
`_v2` uses `scatter_add_` and is much better but has index-shape complexity
bugs under certain configurations — see [02_bugs.md](02_bugs.md#v2-scatter).

---

## 7. Interlaced Convolution (`convo.py` / `partitioning.py`)

### Motivation
The ND-LAr pixel response tensor represents the induced current at a pixel
for an electron landing at each of `nimperpix × nimperpix = 10×10 = 100`
impact positions within the pixel pitch.  A direct convolution of the charge
grid with the response must account for this sub-pixel structure.

The approach is called "interlaced" convolution:
- The charge grid has a fine spacing (1/10 pixel pitch).
- The response tensor also lives on this fine grid but only has support at
  every 10th point (the "lace" spacing = `[10, 10, 1]`).
- For each (impact_x, impact_y) combination (50 pairs exploiting mirror
  symmetry), extract the corresponding "lace" from both signal and response
  and perform a standard FFT convolution.
- Sum the 50 partial convolutions to get the total induced current.

### interlaced_symm_v2 (convo.py:344) — current recommended version

1. **DFT shape**: compute the padded size needed for linear (non-circular)
   convolution: `c_shape = signal_size/lacing + response_size/lacing - 1`.
2. **De-interlacing** (`partitioning.py:63`): for each of the
   `nimperpix/2 = 5` symmetric pairs along the transverse axis, extract
   signal and response laces via strided slicing.  Use the complex-number
   trick: pack a pair `(lace_fwd, flip(lace_rev))` into real and imaginary
   parts of a complex tensor so one FFT handles both.
3. **FFT, multiply, accumulate**: pad to `o_shape`, `fftn`, multiply
   `response_fft × signal_fft`, accumulate into `meas`.
4. **iFFT and unpack**: `ifftn`, then `meas.real + flip(meas.imag)` unpacks
   the symmetric pair trick.

### Recomputing response FFT every call
The line `torch.fft.fftn(res_[None,...], dim=dims)` (`convo.py:385`) is
inside the per-lace loop AND called on every `forward()` invocation.  The
response never changes between calls; pre-computing and caching its FFT
would be a significant speedup.  See [03_gpu_efficiency.md](03_gpu_efficiency.md#fft-cache).

---

## 8. Readout (`readout.py`)

### What it does
Models the pixel electronics chain:
1. Compute a cumulative sum of the induced current waveform (`Xacc = X.cumsum`).
2. Iteratively find the first time tick where the accumulated charge exceeds
   the pixel threshold (discriminator crossing).
3. Record: crossing time, charge at ADC hold time (`hold_t = cross_t + adc_hold_delay`).
4. Reset the accumulator at `hold_t + csa_reset_time`, advance the "start"
   pointer to `hold_t + adc_down_time` to model dead time.
5. Repeat until no more crossings.

The threshold can be per-pixel (loaded from HDF5, shape (N_pixels,)).

Output: flat list of (pixel_x, pixel_y, t_cross, t_hold, t_start) and
corresponding charge.

### Iteration on GPU
The while-loop at `readout.py:67` iterates until `triggered.any() == False`.
Each `triggered.any()` forces a CPU-GPU sync (`.any()` is a reduction that
requires pulling a scalar back to the host).  The number of iterations equals
the maximum number of ADC triggers any single pixel fires.  This can be
many tens of iterations for high-occupancy events.
See [03_gpu_efficiency.md](03_gpu_efficiency.md#readout-loop).

---

## 9. Response Loading (`response.py`)

### What it does
Loads the ND-LAr field response `.npy` file (shape `(45, 45, 6400)` — quarter
of the full response), applies quadrant symmetry (`quadrant_copy`) to produce
the full `(90, 90, 6400)` tensor, then reorders axes to make impact-position
indexing contiguous for the laced convolution:

```
raw (45,45,6400) → full (90,90,6400)
→ view (9, 10, 9, 10, 6400)   # npxl × nimp × npxl × nimp × Nt
→ flip on dims (0,2)
→ reshape (90, 90, 6400)
→ .contiguous()
```

The `view + reshape + contiguous()` sequence creates a full physical copy of
the 90×90×6400 response (≈ 207 MB at float32).  This lives on the GPU for
the entire run.

---

## 10. IO and Loaders (`loaders.py`, `io_nd.py`)

### StepLoader / steps_from_ndh5
Reads steps from an HDF5 file as NumPy arrays via h5py, converts to
`torch.tensor`.  Conversion uses `torch.tensor(np_array, dtype=..., requires_grad=False)`
which always copies.  `torch.from_numpy` followed by `.to(dtype)` would be
more memory-efficient for the in-place case.

### NpzFile / HdfFile
Both use `torch.tensor(data, dtype=..., requires_grad=False)` on every key
access, meaning each item is copied on load.  The HDF5 loader stores an open
file handle (`self._fp = h5py.File(path)`) but does not close it explicitly.

### CustomNDLoader / batch samplers
`io_nd.py` provides custom PyTorch DataLoader subclasses with three sampling
strategies (sorted, eager, lazy).  `SortedLabelBatchSampler` sorts by event
label to keep related steps in the same batch — this is useful to keep steps
belonging to one physics event together, but it discards stochastic shuffling
that might help GPU utilisation.
