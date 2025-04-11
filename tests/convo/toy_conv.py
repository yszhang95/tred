import torch
from torch.fft import fft, ifft, rfft, irfft, fftn, ifftn, rfftn, irfftn
import matplotlib.pyplot as plt

# rfft, two fft -> real+img fft

import torch.nn.functional as F

import numpy as np

def do_rfft1d(s, r):
    '''
    s is batched;
    r is unbatched;
    '''
    ns = s.size(1)
    nr = r.size(0)
    n = ns+nr-1
    sig = rfft(s, n=n)
    res = rfft(r, n=n)
    return sig, res, n

def rconv1d(s, r):
    '''
    s is batched;
    r is unbatched;
    '''
    sig, res, n = do_rfft1d(s, r)
    return irfft(sig*res.unsqueeze(0), n=n)

def do_fft1d(s, r):
    '''
    s is batched;
    r is unbatched;
    '''
    ns = s.size(1)
    nr = r.size(0)
    sig = fft(s, n=ns+nr-1)
    res = fft(r, n=ns+nr-1)
    return sig, res

def cconv1d(s, r):
    '''
    s is batched;
    r is unbatched;
    '''
    sig, res = do_fft1d(s, r)
    return ifft(sig*res.unsqueeze(0)).real

def cconv1d_loop(s, r):
    nb = s.shape[0]
    ns = s.shape[1]  # spatial dims
    nr = r.shape[0]      # kernel dims
    n = ns+nr-1
    o = []
    res = fft(r, n=n)
    for i in range(nb):
        sig = fft(s[i], n=n)
        oo = ifft(sig * res, n=n)
        o.append(oo)
    return torch.stack(o, dim=0).real

def do_split_fft1d(s, r):
    '''
    s is batched;
    r is unbatched;
    '''
    nb = s.size(0)
    ns = s.size(1)
    nr = r.size(0)
    s0, s1 = s[:nb//2], s[nb//2:]
    sig = fft(torch.complex(s0,s1), n=ns+nr-1)
    res = fft(r, n=ns+nr-1)
    return sig, res

def split_conv1d(s, r):
    '''
    s is batched;
    r is unbatched;
    '''
    sig, res = do_split_fft1d(s, r)
    output = ifft(sig*res.unsqueeze(0))
    return torch.vstack([output.real, output.imag])

def perf_conv1d_speed(s, r, f):

    # perf
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    torch.cuda.synchronize()
    f(s,r)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    # print(f"Execution time: {elapsed_time_ms:.4f} ms for", str(f))
    return elapsed_time_ms

def perf_conv1d_memory(s, r, f):
    # perf
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    f(s,r)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated()/1024**2
    # print('torch peak memory [MB] for input', mem, 'for', str(f))
    return mem

def test_conv():
    torch.manual_seed(10)
    torch.cuda.reset_peak_memory_stats()
    nbatches = 10
    ns = 500
    nr = 1000
    s = torch.rand(ns).repeat(nbatches).view(nbatches, ns).to(torch.float64).to('cuda')
    r = torch.rand(nr).to(torch.float64).to('cuda')

    # test correctness
    o1 = rconv1d(s,r)
    o2 = cconv1d(s,r)
    o3 = split_conv1d(s,r)

    scpu = s.cpu()[0]
    rcpu = r.cpu()
    o1cpu = o1.cpu()
    o2cpu = o2.cpu()
    o3cpu = o3.cpu()
    o1np = o1cpu[0].numpy()
    o2np = o2cpu[0].numpy()
    o3np = o3cpu[0].numpy()

    # test correctness
    onp = np.convolve(scpu.numpy(), rcpu.numpy()).astype(np.float64)
    assert np.allclose(o2np, o3np, atol=1E-14, rtol=1E-12)
    assert np.allclose(o1np, o3np, atol=1E-14, rtol=1E-12)
    assert np.allclose(o3np, onp, atol=1E-14, rtol=1E-12)

def benchmark_conv_across_batches(batch_sizes, ns, nr, device='cuda'):
    times = {'rconv1d': [], 'cconv1d': [], 'split_conv1d': [], 'cconv1d_loop': []}
    mems = {'rconv1d': [], 'cconv1d': [], 'split_conv1d': [], 'cconv1d_loop': []}
    meminputs = []

    torch.manual_seed(10)

    for nb in batch_sizes:
        print(f"Testing nbatches = {nb}")
        s = torch.rand((nb, ns), device=device)
        # s = torch.rand(ns).repeat(nb).view(nb, ns).to(torch.float32).to(device)
        r = torch.rand(nr, device=device)

        for name, fn in zip(['rconv1d', 'cconv1d', 'split_conv1d', 'cconv1d_loop'],
                            [rconv1d, cconv1d, split_conv1d, cconv1d_loop]):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # Measure time
            m = perf_conv1d_memory(s, r, fn)
            perf_conv1d_speed(s, r, fn)
            t = perf_conv1d_speed(s, r, fn)

            times[name].append(t)
            mems[name].append(m)

        # Estimate input memory in bytes
        s_mem = (nb+1) * (ns+nr-1) * s.element_size()
        meminputs.append(s_mem  / 1024**2)

    return times, mems, meminputs

def plot_perf_vs_batches(batch_sizes, times, mems, meminputs, shape_s, shape_r, figname):
    plt.figure(figsize=(12, 5))

    # Time Plot
    plt.subplot(1, 2, 1)
    for name, vals in times.items():
        if name == 'cconv1d_loop':
            plt.plot(batch_sizes[:3], vals[:3], 'o-', label=name)
        else:
            plt.plot(batch_sizes, vals, 'o-', label=name)
    plt.xlabel('Number of Batches')
    plt.ylabel('Execution Time (ms)')
    plt.title('Convolution Runtime vs Batch Size')
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    ax.text(0.25, 0.65, f"Signal shape {tuple(shape_s)}\nResponse shape {tuple(shape_r)}",
            transform=ax.transAxes, fontsize=12, horizontalalignment='center', verticalalignment='center')

    # Memory Plot
    plt.subplot(1, 2, 2)
    for name, vals in mems.items():
        plt.plot(batch_sizes, vals, 'o-', label=name)
    plt.plot(batch_sizes, meminputs, 'o-', label='Memory for padded input')
    plt.xlabel('Number of Batches')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Memory Usage vs Batch Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig(figname)

def main():
    test_conv()

    for ns, nr in zip([500,], [6000,]):
        batch_sizes = [50, 100, 500, 1_000, 5_000, 10_000, 20_000, 50_000, 75_000, 100_000]

        size_str = f'sshape_{ns}_rshape_{nr}'

        times, mems, meminputs = benchmark_conv_across_batches(batch_sizes, ns, nr)
        plot_perf_vs_batches(batch_sizes, times, mems, meminputs, shape_s=[ns,], shape_r=[nr,], figname=f'conv1d_resources_{size_str}.png')

if __name__ == '__main__':
    main()
