import torch
from torch.fft import fft2, ifft2, rfft2, irfft2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def do_rfft2d(s, r):
    ns = s.shape[1:]  # spatial dims
    nr = r.shape      # kernel dims
    n = [ns[i] + nr[i] - 1 for i in range(2)]
    sig = rfft2(s, s=n, dim=(1, 2))
    res = rfft2(r, s=n)
    return sig, res, n

def rconv2d(s, r):
    sig, res, n = do_rfft2d(s, r)
    return irfft2(sig * res[None, ...], s=n, dim=(1, 2))

def do_fft2d(s, r):
    ns = s.shape[1:]
    nr = r.shape
    n = [ns[i] + nr[i] - 1 for i in range(2)]
    sig = fft2(s, s=n, dim=(1, 2))
    res = fft2(r, s=n)
    return sig, res

def cconv2d(s, r):
    sig, res = do_fft2d(s, r)
    return ifft2(sig * res[None, ...], dim=(1, 2)).real

def cconv2d_loop(s, r):
    nb = s.shape[0]
    ns = s.shape[1:]
    nr = r.shape
    n = [ns[i] + nr[i] - 1 for i in range(2)]
    o = []
    res = fft2(r, s=n)
    for i in range(nb):
        sig = fft2(s[i], s=n)
        oo = ifft2(sig * res, s=n)
        o.append(oo)
    return torch.stack(o, dim=0).real

def do_split_fft2d(s, r):
    nb = s.shape[0]
    s0, s1 = s[:nb//2], s[nb//2:]
    s_complex = torch.complex(s0, s1)
    ns = s0.shape[1:]
    nr = r.shape
    n = [ns[i] + nr[i] - 1 for i in range(2)]
    sig = fft2(s_complex, s=n, dim=(1, 2))
    res = fft2(r, s=n)
    return sig, res

def split_conv2d(s, r):
    sig, res = do_split_fft2d(s, r)
    out = ifft2(sig * res[None, ...], dim=(1, 2))
    return torch.cat([out.real, out.imag], dim=0)

def perf_conv2d_speed(s, r, f):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    torch.cuda.synchronize()
    f(s, r)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)

def perf_conv2d_memory(s, r, f):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    f(s, r)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

def benchmark_conv2d_across_batches(batch_sizes, shape_s, shape_r, device='cuda'):
    times = {'rconv2d': [], 'cconv2d': [], 'split_conv2d': [], 'cconv2d_loop': []}
    mems = {'rconv2d': [], 'cconv2d': [], 'split_conv2d': [], 'cconv2d_loop': []}
    meminputs = []

    for nb in batch_sizes:
        print(f"Testing nbatches = {nb}")
        s = torch.rand((nb, *shape_s), device=device)
        r = torch.rand(shape_r, device=device)

        for name, fn in zip(['rconv2d', 'cconv2d', 'split_conv2d', 'cconv2d_loop'],
                            [rconv2d, cconv2d, split_conv2d, cconv2d_loop]):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            mem = perf_conv2d_memory(s, r, fn)
            perf_conv2d_speed(s, r, fn)
            time = perf_conv2d_speed(s, r, fn)

            times[name].append(time)
            mems[name].append(mem)

        o1 = cconv2d_loop(s, r)
        o2 = cconv2d(s, r)
        assert torch.allclose(o1, o2, atol=1e-5)

        input_bytes = (nb + 1) * np.prod([shape_s[i] + shape_r[i] - 1 for i in range(2)]) * s.element_size()
        meminputs.append(input_bytes / 1024**2)

    return times, mems, meminputs

def plot_perf_vs_batches(batch_sizes, times, mems, meminputs, shape_s, shape_r, figname):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, vals in times.items():
        plt.plot(batch_sizes, vals, 'o-', label=name)
    plt.xlabel('Number of Batches')
    plt.ylabel('Execution Time (ms)')
    plt.title('2D Convolution Runtime vs Batch Size')
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    ax.text(0.25, 0.65, f"Signal shape {tuple(shape_s)}\nResponse shape {tuple(shape_r)}",
            transform=ax.transAxes, fontsize=12, horizontalalignment='center', verticalalignment='center')

    plt.subplot(1, 2, 2)
    for name, vals in mems.items():
        plt.plot(batch_sizes, vals, 'o-', label=name)
    plt.plot(batch_sizes, meminputs, 'o-', label='Memory for padded input')
    plt.xlabel('Number of Batches')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('2D Memory Usage vs Batch Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(figname)
    plt.show()


def test_conv2d():
    torch.manual_seed(10)
    torch.cuda.reset_peak_memory_stats()
    nbatches = 10
    ns = (10, 10)
    nr = (5, 5)
    s = torch.rand(ns).repeat(nbatches, 1, 1).view(nbatches, *ns).to(torch.float64).to('cuda')
    r = torch.rand(nr).to(torch.float64).to('cuda')
    nsp = (ns[i] + nr[i]*2 - 2 for i in range(2))

    o = F.conv2d(F.pad(s, (4, 4, 4, 4,), 'constant', 0).view(nbatches, 1, *nsp), r.flip(dims=(0,1)).view(1, 1, *nr), padding='valid')
    o = o.cpu()

    # test correctness
    o1 = rconv2d(s,r).cpu()
    o2 = cconv2d(s,r).cpu()
    o3 = split_conv2d(s,r).cpu()

    assert torch.allclose(o, o1)
    assert torch.allclose(o, o2)
    assert torch.allclose(o, o3)


def main():
    test_conv2d()

    batch_sizes = [10, 50, 100, 200, 500, 1000, 2000]

    for shape_s, shape_r in zip([(5, 500),], [(9, 6400),]):

        size_str = f'sshape_{"_".join(map(str, shape_s))}_rshape_{"_".join(map(str, shape_r))}'

        times, mems, meminputs = benchmark_conv2d_across_batches(batch_sizes, shape_s, shape_r)
        plot_perf_vs_batches(batch_sizes, times, mems, meminputs, shape_s, shape_r, f'conv2d_resources_{size_str}.png')

        print(f"Finished plot conv2d_resources_{size_str}.png")

if __name__ == '__main__':
    main()
