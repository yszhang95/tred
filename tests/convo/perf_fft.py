import numpy as np
import torch
import matplotlib.pyplot as plt

def prune_fft1d_np(x, P: int, N: int):
    """
    NumPy version of `prune_fft1d`.

    Parameters
    ----------
    x : np.ndarray
        Shape (L,) or (B, L). Real or complex.  Will be promoted to complex64.
    P : int
        Sub‑FFT length.  Must divide N.
    N : int
        Desired full‑length FFT (N % P == 0).

    Returns
    -------
    np.ndarray
        Shape (N,) if input was 1‑D, else (B, N).
    """
    # Ensure ndarray
    x = np.asarray(x)
    is_batched = x.ndim > 1
    if not is_batched:
        x = x[np.newaxis, :]
    if x.ndim > 2:
        raise ValueError("`x` must be 1‑D or 2‑D.")

    B, L = x.shape
    xfull = np.zeros((B, P), dtype=np.complex64)
    xfull[:, :L] = x.astype(np.complex64, copy=False)

    if N % P:
        raise ValueError("N must be an integer multiple of P.")
    Q = N // P

    # Expand to (B, Q, P); copy() because broadcast_to returns a view
    xfull = np.broadcast_to(xfull[:, None, :], (B, Q, P)).copy()

    ind = np.arange(P, dtype=np.float32)  # shared across all batches

    for i in range(Q):
        twiddle = np.exp(-1j * 2 * np.pi / N * ind * i, dtype=np.complex64)
        xfull[:, i, :] *= twiddle
        xfull[:, i, :] = np.fft.fft(xfull[:, i, :], n=P, axis=-1)

    # Re‑interleave: (B, P, Q) → (B, N)
    xfull = xfull.transpose(0, 2, 1).reshape(B, N)

    return xfull.squeeze(0) if not is_batched else xfull

def test_prune_fft1d_np():
    x = np.array([1, 2, 3], dtype=np.float32)
    assert np.all(np.absolute(prune_fft1d_np(x, P=4, N=8) - np.fft.fft(x, n=8)) < 1E-5)


def prune_fft1d(x, P, N):
    '''
    x is batched or not
    '''
    isbatched = x.ndim > 1
    if not isbatched:
        x = x.unsqueeze(0)
    if x.ndim > 2:
        raise ValueError
    xfull = torch.zeros((x.size(0), P), dtype=torch.complex64, device=x.device)
    xfull[:,:x.size(1)] = x
    # print(xfull)
    Q = N//P
    if N % P != 0:
        raise ValueError
    # enforce the contiguous memory layout; otherwise in-place operations give wrong results
    xfull = xfull.view(x.size(0), 1, P).expand(x.size(0), Q, P).clone().detach()

    # for i in range(Q):
    #     ind = torch.arange(P)
    #     xfull[:,i,:] *= torch.exp(-1j * torch.pi * 2 / N * ind * i).unsqueeze(0)
    #     xfull[:,i,:] = torch.fft.fft(xfull[:,i,:], n=P, dim=-1)
    twiddle = torch.exp(-2j*torch.pi/N * torch.arange(Q).unsqueeze(1) * torch.arange(P).unsqueeze(0))
    twiddle = twiddle.to(x.device)
    xfull = torch.fft.fft(xfull.mul_(twiddle))
    xfull = xfull.permute((0,2,1)).contiguous().view(x.size(0), N)
    if not isbatched:
        xfull = xfull.squeeze(0)
    return xfull

def test_prune_fft1d():
    x  = torch.tensor([1, 2, 3], dtype=torch.complex64)
    P, N = 4, 8

    xf_pruned = prune_fft1d(x, P, N)
    xf_full   = torch.fft.fft(torch.nn.functional.pad(x, (0, N - x.numel())), n=N)
    assert (xf_pruned - xf_full).abs().max() < 1E-5


def td_fft3d(x, P, N):
    """
    3‑D transform–decomposition FFT for inputs that are *non‑zero only in
    the block 0≤n_i<P_i*.  Works on real or complex tensors.

    Parameters
    ----------
    x : (P1,P2,P3) tensor  – the non‑zero spatial block
    N : tuple (N1,N2,N3)  – full transform length in each dim
    P : tuple (P1,P2,P3)  – block length in each dim (must divide N)

    Returns
    -------
    X : (N1,N2,N3) complex tensor – full DFT of the zero‑padded volume
    """
    device = x.device
    dtype  = torch.complex64
    N1,N2,N3 = N
    P1,P2,P3 = P
    if not (N1%P1==0 and N2%P2==0 and N3%P3==0):
        raise ValueError("P must divide N")
    Q1,Q2,Q3 = N1//P1, N2//P2, N3//P3

    # 1) replicate the active block
    xfull = x.to(dtype).expand(Q1,Q2,Q3,P1,P2,P3).clone().detach()

    # 2) twiddle factors  (broadcast shapes shown as comments)
    n1 = torch.arange(P1, device=device)   # (P1)
    a1 = torch.arange(Q1, device=device)   # (Q1)
    tw1 = torch.exp(-2j*torch.pi/N1 * n1[None,None,None,:,None,None]
                                   * a1[:,None,None,None,None,None])

    n2 = torch.arange(P2, device=device)
    a2 = torch.arange(Q2, device=device)
    tw2 = torch.exp(-2j*torch.pi/N2 * n2[None,None,None,None,:,None]
                                   * a2[None,:,None,None,None,None])

    n3 = torch.arange(P3, device=device)
    a3 = torch.arange(Q3, device=device)
    tw3 = torch.exp(-2j*torch.pi/N3 * n3[None,None,None,None,None,:]
                                   * a3[None,None,:,None,None,None])

    xfull.mul_(tw1 * tw2 * tw3)

    # 3) local P1×P2×P3 FFT
    Xsub = torch.fft.fftn(xfull, s=(P1,P2,P3), dim=(-3,-2,-1))

    # 4) shuffle axes into (N1,N2,N3)
    X = (Xsub
         .permute(3,0,4,1,5,2)         # (P1,Q1,P2,Q2,P3,Q3)
         .reshape(N1, N2, N3))
    return X

def td_fft3d_py(x, P : tuple[int, int, int], N : tuple[int, int, int]):
    """
    3‑D transform–decomposition FFT for inputs that are *non‑zero only in
    the block 0≤n_i<P_i*.  Works on real or complex tensors.

    Parameters
    ----------
    x : (P1,P2,P3) tensor  – the non‑zero spatial block
    N : tuple (N1,N2,N3)  – full transform length in each dim
    P : tuple (P1,P2,P3)  – block length in each dim (must divide N)

    Returns
    -------
    X : (N1,N2,N3) complex tensor – full DFT of the zero‑padded volume
    """
    device = x.device
    dtype  = torch.complex64
    N1,N2,N3 = N
    P1,P2,P3 = P
    if not (N1%P1==0 and N2%P2==0 and N3%P3==0):
        raise ValueError("P must divide N")
    Q1,Q2,Q3 = N1//P1, N2//P2, N3//P3
    Q = Q1, Q2, Q3

    # 1) replicate the active block
    xfull = x.to(dtype).view(x.size(0), 1,1,1, P1, P2, P3).expand(-1, Q1,Q2,Q3,P1,P2,P3).clone().detach()

    # 2) twiddle factors  (broadcast shapes shown as comments)
    pi = 3.141592653589793
    n1 = torch.arange(P1, dtype=torch.float32, device=device)   # (P1)
    a1 = torch.arange(Q1, dtype=torch.float32, device=device)   # (Q1)
    theta1 = (-2*torch.pi/N1 * n1[None,None,None,:,None,None]
                                   * a1[:,None,None,None,None,None])
    tw1 = torch.complex(torch.cos(theta1), torch.sin(theta1))

    n2 = torch.arange(P2, dtype=torch.float32, device=device)
    a2 = torch.arange(Q2, dtype=torch.float32, device=device)
    theta2 = (-2*torch.pi/N2 * n2[None,None,None,None,:,None]
                                   * a2[None,:,None,None,None,None])
    tw2 = torch.complex(torch.cos(theta2), torch.sin(theta2))

    n3 = torch.arange(P3, device=device)
    a3 = torch.arange(Q3, device=device)
    theta3 = (-2*torch.pi/N3 * n3[None,None,None,None,None,:]
                                   * a3[None,None,:,None,None,None])
    tw3 = torch.complex(torch.cos(theta3), torch.sin(theta3))

    xfull.mul_((tw1 * tw2 * tw3).unsqueeze(0))

    # 3) local P1×P2×P3 FFT
    Xsub = torch.fft.fftn(xfull, s=(P1,P2,P3), dim=(-3,-2,-1))

    # 4) shuffle axes into (N1,N2,N3)
    X = (Xsub
         .permute(0,4,1,5,2,6,3)         # (P1,Q1,P2,Q2,P3,Q3)
         .reshape(x.size(0), N1, N2, N3))
    return X

td_fft3d_jit = torch.jit.script(td_fft3d_py)


def test_td_fft3d():

    P = (8, 8, 8)
    N = (32, 32, 32)

    torch.manual_seed(16)
    sig = 1+torch.randn(*P)

    y_td   = td_fft3d(sig, P, N)          # just the P‑block
    y_full = torch.fft.fftn(sig, s=N, dim=(-3,-2,-1))

    # assert torch.allclose(y_td, y_full)
    assert torch.all((y_td - y_full).abs() < 1E-4)


def td_fftn_nd(x_block, P, N, dtype=torch.complex64):
    """
    N‑D transform–decomposition FFT for a signal whose non‑zero samples are
    confined to the hyper‑block  0 ≤ n_i < P_i  in every spatial dimension.

    Parameters
    ----------
    x_block : (..., *P1, P2, …, PD*) tensor
        *Last D axes* hold the active block.  Any number of batch axes may
        precede them.
    N : tuple/list of ints (N1, …, ND)
        Full FFT length in each spatial dimension.
    P : tuple/list of ints (P1, …, PD)
        Block length in each spatial dimension (must divide N element‑wise).

    Returns
    -------
    X : (..., *N1, N2, …, ND*)  tensor (complex64)
        FFT of the zero‑padded volume.
    """
    D = len(N)
    assert len(P) == D, "N and P must have the same length"
    Bshape = x_block.shape[:-D]        # batch prefix
    device = x_block.device

    # ------------------------------------------------------------------ #
    # 1) make a view of shape  (B, Q1, …, QD, P1, …, PD)
    # ------------------------------------------------------------------ #
    Q = [Ni // Pi for Ni, Pi in zip(N, P)]
    for i, (Ni, Pi) in enumerate(zip(N, P)):
        if Ni % Pi != 0:
            raise ValueError(f"P[{i}]={Pi} does not divide N[{i}]={Ni}")

    view_shape = (*Bshape, *Q, *P)                 # e.g. (B,Q1,Q2,P1,P2)
    # 1) Add singleton axes for Q (replication dims) before each P axis
    #    Example: (B, P1, P2, P3) → (B, 1, 1, 1, P1, P2, P3) for D=3
    for i in range(len(P)):
        x_block = x_block.unsqueeze(len(Bshape) + i)
    xfull = x_block.to(dtype).expand(view_shape).clone().detach()

    # ------------------------------------------------------------------ #
    # 2) multiply by product of twiddle factors
    # ------------------------------------------------------------------ #
    twiddle = 1.0 + 0.0j
    for axis, (Ni, Pi, Qi) in enumerate(zip(N, P, Q)):
        # axis index in xfull   ───►   batch … Qi‑axis … Pi‑axis …
        qi = len(Bshape) + axis
        pi = len(Bshape) + D + axis
        a  = torch.arange(Qi, device=device).view(
                 *([1]*(qi) + [Qi] + [1]* (xfull.ndim - qi - 1)))
        b  = torch.arange(Pi, device=device).view(
                 *([1]*(pi) + [Pi] + [1]*(xfull.ndim - pi - 1)))
        twiddle = twiddle * torch.exp(-2j*torch.pi/Ni * a * b)

    xfull.mul_(twiddle)

    # ------------------------------------------------------------------ #
    # 3) local P1×···×PD FFT over the last D axes
    # ------------------------------------------------------------------ #
    last_axes = tuple(range(-D, 0))
    Xsub = torch.fft.fftn(xfull.view(-1,*P), s=P, dim=last_axes)
    Xsub = Xsub.view(*Bshape, *Q, *P)

    # ------------------------------------------------------------------ #
    # 4) interleave (b_i, a_i) so that k_i = a_i + Qi * b_i
    #    The permutation is:  (batch, b1, a1, b2, a2, …, bD, aD)
    # ------------------------------------------------------------------ #
    perm = list(range(len(Bshape)))                # batch axes unchanged
    for i in range(D):
        perm.extend([len(Bshape) + D + i,          # b_i  (Pi‑axis)
                     len(Bshape) + i])             # a_i  (Qi‑axis)
    X = Xsub.permute(perm).reshape(*Bshape, *N)
    return X

# -------------------------------------------------------------------- #
# Example + correctness check
# -------------------------------------------------------------------- #
def test_td_fftnd():
    torch.manual_seed(16)

    # 2‑batch, 4‑D signal:  (B, P1,P2,P3,P4)   sparsity block 6×4×3×5
    B     = 2
    P     = (6, 4, 3, 5)
    N     = (24, 16, 12, 20)
    signal_block = 1+torch.randn(B, *P).to(torch.complex64) + torch.randn(B, *P).to(torch.complex64) * 1j

    # TD result
    X_td = td_fftn_nd(signal_block, P, N)

    # Reference (pad + full fftn, *slow but correct*)
    pad = torch.zeros(B, *N).to(torch.complex128)
    slices = (slice(None),) + tuple(slice(0, Pi) for Pi in P)
    pad[slices] = signal_block
    X_ref = torch.fft.fftn(pad, s=N, dim=tuple(range(-4,0)))

    err_max = (X_td - X_ref).abs().max().item()

    assert err_max < 1E-4

def benchmark_fft(fft_func, x):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(x.device)

    # Warm-up
    for _ in range(4):
        _ = fft_func(x)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fft_func(x)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)

    stmt = "fft_func(x)"
    globals_dict = {"x": x, "fft_func" : fft_func}

    t = benchmark.Timer(stmt, globals=globals_dict)
    result = t.blocked_autorange(min_run_time=0.1)
    elapsed = result.median * 1E3 # ms

    mem = torch.cuda.max_memory_allocated(x.device) / 1024**2  # MiB

    return elapsed, mem

def make_plot(results, batch_sizes, P, N, figname):
    # Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8*2, 8))

    ax1.plot(batch_sizes, results['pruned']['runtime'], 'o-', label='Pruned FFT Runtime (ms)')
    ax1.plot(batch_sizes, results['full']['runtime'], 'o--', label='Full FFT Runtime (ms)')
    if results.get('pruned_jit', None):
        ax1.plot(batch_sizes, results['pruned_jit']['runtime'], 'o-.', label='Pruned FFT (jit) Runtime (ms)')
    ax1.plot(batch_sizes, results['full']['runtime'], 'o--', label='Full FFT Runtime (ms)')
    ax2.plot(batch_sizes, results['pruned']['memory'], 's-', color='tab:red', label='Pruned FFT CUDA Mem (MiB)')
    ax2.plot(batch_sizes, results['full']['memory'], 's--', color='tab:orange', label='Full FFT CUDA Mem (MiB)')

    ax1.set_xlabel('Batch size')
    ax2.set_xlabel('Batch size')
    ax1.set_ylabel('Runtime (ms)', color='tab:blue')
    ax2.set_ylabel('CUDA Memory (MiB)', color='tab:red')
    ax1.set_title('Runtime vs Batch Size')
    ax2.set_title('CUDA Memory vs Batch Size')

    ax1.text(0.2, 0.5, f"P {P}\nN {N}", transform=ax1.transAxes)

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.savefig(figname)


def perf_td_fft1d():

    # Parameters
    P = 384
    N = 6912
    L = 380

    figname = f'resources_prune_fft1d_P{P}_N{N}.png'

    batch_sizes = [1_000, 10_000, 20_000, 30_000, 50_000, 60_000]

    results = {
        'pruned': {'runtime': [], 'memory': []},
        'full': {'runtime': [], 'memory': []}
    }

    device = torch.device("cuda")

    torch.manual_seed(16)
    for B in batch_sizes:
        x = torch.randn(B, L, dtype=torch.float32, device=device)

        elapsed, mem = benchmark_fft(lambda x : prune_fft1d(x, P, N), x)
        results['pruned']['runtime'].append(elapsed)
        results['pruned']['memory'].append(mem)

        elapsed, mem = benchmark_fft(lambda x: torch.fft.fft(x, n=N, dim=-1), x)
        results['full']['runtime'].append(elapsed)
        results['full']['memory'].append(mem)

    make_plot(results, batch_sizes, P, N, figname)
    results = None

def perf_td_fft2d():

    # Parameters
    P = 8, 384
    N = 16, 6912

    size_str = f'P{"_".join(map(str, P))}_N{"_".join(map(str, N))}'

    figname = f'resources_prune_fft2d_{size_str}.png'

    batch_sizes = [20, 50, 100, 150, 200, 400]

    results = {
        'pruned': {'runtime': [], 'memory': []},
        'full': {'runtime': [], 'memory': []}
    }

    device = torch.device("cuda")

    torch.manual_seed(16)
    for B in batch_sizes:
        x = torch.randn((B, *P), dtype=torch.float32, device=device)

        with torch.no_grad():
            elapsed, mem = benchmark_fft(lambda x : td_fftn_nd(x, P, N), x)
            results['pruned']['runtime'].append(elapsed)
            results['pruned']['memory'].append(mem)

            elapsed2, mem2 = benchmark_fft(lambda x: torch.fft.fftn(x, s=N, dim=tuple(range(-len(N),0))), x)
            results['full']['runtime'].append(elapsed2)
            results['full']['memory'].append(mem2)

    make_plot(results, batch_sizes, P, N, figname)
    results = None

def perf_td_fft3d():

    # Parameters
    P = 8, 8, 384
    N = 16, 16, 6912

    size_str = f'P{"_".join(map(str, P))}_N{"_".join(map(str, N))}'

    figname = f'resources_prune_fft3d_{size_str}.png'

    batch_sizes = [20, 50, 100, 150, 200]

    results = {
        'pruned': {'runtime': [], 'memory': []},
        'pruned_jit': {'runtime': [], 'memory': []},
        'full': {'runtime': [], 'memory': []}
    }

    device = torch.device("cuda")

    torch.manual_seed(16)
    for B in batch_sizes:
        x = torch.randn((B, *P), dtype=torch.float32, device=device)

        elapsed, mem = benchmark_fft(lambda x : td_fftn_nd(x, P, N), x)
        results['pruned']['runtime'].append(elapsed)
        results['pruned']['memory'].append(mem)

        elapsed2, mem2 = benchmark_fft(lambda x : td_fft3d_jit(x, P, N), x)
        results['pruned_jit']['runtime'].append(elapsed2)
        results['pruned_jit']['memory'].append(mem2)

        elapsed3, mem3 = benchmark_fft(lambda x: torch.fft.fftn(x, s=N, dim=tuple(range(-len(N),0))), x)
        results['full']['runtime'].append(elapsed3)
        results['full']['memory'].append(mem3)


    make_plot(results, batch_sizes, P, N, figname)
    results = None


def profile_prune_fft3d():
    # Parameters
    P = 8, 8, 384
    N = 16, 16, 6912
    # P =  2, 2, 128
    # N = 4, 4, 128*128*2

    # size_str = f'P{"_".join(map(str, P))}_N{"_".join(map(str, N))}'

    # figname = f'resources_prune_fft3d_{size_str}.png'

    B = 128

    device = torch.device("cuda")
    x = torch.randn((B, *P), dtype=torch.float32, device=device, requires_grad=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(x.device)
    full_dims = tuple(range(-len(N),0,1))
    for _ in range(5):
        torch.fft.fftn(x, s=N, dim=full_dims)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof_full:
        torch.fft.fftn(x, s=N, dim=full_dims)
        torch.cuda.synchronize()

    print("Peak CUDA memory", torch.cuda.max_memory_allocated()/1024**2, "MB")

    print(prof_full.key_averages().table(
        sort_by="cuda_time_total", row_limit=10, header="Full FFT"))


    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(x.device)

    # Warm-up
    for _ in range(3):
        td_fftn_nd(x, P, N)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof_pruned:
        td_fftn_nd(x, P, N)
        torch.cuda.synchronize()

    print("Peak CUDA memory", torch.cuda.max_memory_allocated()/1024**2, "MB")

    print(prof_pruned.key_averages().table(
        sort_by="cuda_time_total", row_limit=15, header="Pruned FFT"))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(x.device)
    for _ in range(3):
        td_fft3d_py(x, P, N)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof_pruned_cos_sin:
        td_fft3d_py(x, P, N)
        torch.cuda.synchronize()

    print("Peak CUDA memory", torch.cuda.max_memory_allocated()/1024**2, "MB")

    print(prof_pruned_cos_sin.key_averages().table(
        sort_by="cuda_time_total", row_limit=15, header="Pruned FFT (cos+isin)"))


import torch
import torch.utils.benchmark as benchmark
from collections import Counter

# Prime factorization utilities
def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 1
    if n > 1:
        factors.append(n)
    return dict(Counter(factors))

def format_factors(n):
    factors = prime_factors(n)
    return " × ".join([f"{p}" if e == 1 else f"{p}^{e}" for p, e in sorted(factors.items())])

# Benchmark function
def benchmark_fftn():
    def call_fftn(length, device='cuda'):
        shape = (4, 4, length)
        total_elements = 4 * 4 * length
        batch = 2**20 // total_elements
        batch = max(1, batch)
        size = (batch, *shape)

        x = torch.randn((*size, 2), dtype=torch.float32, device=device)
        x_complex = torch.view_as_complex(x)

        stmt = "torch.fft.fftn(x_complex, dim=(-3, -2, -1))"
        globals_dict = {"x_complex": x_complex, "torch": torch}

        t = benchmark.Timer(stmt, globals=globals_dict)
        result = t.blocked_autorange(min_run_time=0.1)

        arraystr = f"{batch:<5d}×4×4×{length:<5d}"

        total = batch * 4 * 4 * length
        print(f"{length:<7d} | {format_factors(length):<20s} | "
              f"{arraystr:<20s}| {total:<9d} | {result.median * 1e6:.3f} us")
        return result

    # Header
    print(f"{'Length':<7s} | {'Factors':<20s} | {'Shape':<20s} | {'Total':<9s} | {'Time':<10s}")
    print("-" * 80)

    # Run benchmarks
    for length in [4, 9, 16, 128, 128*3, 1024, 960, 800, 896, 1000, 6000, 8000, 8096, 8192]:
        call_fftn(length)
    print()

# Benchmark function
def benchmark_fft1d():
    def call_fft1d(length, device='cuda'):
        shape = (length,)
        total_elements = length
        batch = 2**20 // total_elements
        batch = max(1, batch)
        size = (batch, *shape)

        x = torch.randn((*size, 2), dtype=torch.float32, device=device)
        x_complex = torch.view_as_complex(x)

        stmt = "torch.fft.fftn(x_complex, dim=-1)"
        globals_dict = {"x_complex": x_complex, "torch": torch}

        t = benchmark.Timer(stmt, globals=globals_dict)
        result = t.blocked_autorange(min_run_time=0.1)

        arraystr = f"{batch:<8d}×{length:<5d}"

        total = batch *  length
        print(f"{length:<7d} | {format_factors(length):<20s} | "
              f"{arraystr:<20s}| {total:<9d} | {result.median * 1e6:.3f} us")
        return result

    # Header
    print(f"{'Length':<7s} | {'Factors':<20s} | {'Shape':<20s} | {'Total':<9s} | {'Time':<10s}")
    print("-" * 80)

    # Run benchmarks
    for length in [4, 9, 16, 128, 1024, 960, 800, 896, 1000, 6000, 8000, 8096, 8192]:
        call_fft1d(length)
    print()


def main():
    test_prune_fft1d_np()
    test_prune_fft1d()
    test_td_fft3d()
    test_td_fftnd()
    perf_td_fft1d()
    perf_td_fft2d()
    perf_td_fft3d()
    profile_prune_fft3d()
    benchmark_fftn()
    benchmark_fft1d()

if __name__ == '__main__':
    main()
