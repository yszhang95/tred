import torch
from torch.fft import fftn, ifftn, rfftn, irfftn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def do_rfft3d(s, r):
    ns = s.shape[1:]  # spatial dims
    nr = r.shape      # kernel dims
    n = [ns[i] + nr[i] - 1 for i in range(3)]
    sig = rfftn(s, s=n, dim=(1, 2, 3))
    res = rfftn(r, s=n)
    return sig, res, n

def rconv3d(s, r):
    sig, res, n = do_rfft3d(s, r)
    return irfftn(sig * res[None, ...], s=n, dim=(1, 2, 3))

def do_fft3d(s, r):
    ns = s.shape[1:]
    nr = r.shape
    n = [ns[i] + nr[i] - 1 for i in range(3)]
    nsz = [[nr[i] - 1, 0] for i in range(3)]
    nsz = [j for i in nsz[::-1] for j in i[::-1]]
    nsr = [[ns[i] - 1, 0] for i in range(3)]
    nsr = [j for i in nsr[::-1] for j in i[::-1]]
    sig = fftn(F.pad(s, nsz), dim=(1, 2, 3)).contiguous()
    res = fftn(F.pad(r, nsr), ).contiguous()
    return sig, res

def cconv3d(s, r):
    sig, res = do_fft3d(s, r)
    return ifftn(sig * res[None, ...], dim=(1, 2, 3)).real

def cconv3d_loop(s, r):
    nb = s.shape[0]
    ns = s.shape[1:]  # spatial dims
    nr = r.shape      # kernel dims
    n = [ns[i] + nr[i] - 1 for i in range(3)]
    o = []
    res = fftn(r, s=n)
    for i in range(nb):
        sig = fftn(s[i], s=n)
        oo = ifftn(sig * res, s=n)
        o.append(oo)
    return torch.stack(o, dim=0).real

def do_split_fft3d(s, r):
    nb = s.shape[0]
    s0, s1 = s[:nb//2], s[nb//2:]
    s_complex = torch.complex(s0, s1)
    ns = s0.shape[1:]
    nr = r.shape
    n = [ns[i] + nr[i] - 1 for i in range(3)]
    sig = fftn(s_complex, s=n, dim=(1, 2, 3))
    res = fftn(r, s=n)
    return sig, res

def split_conv3d(s, r):
    sig, res = do_split_fft3d(s, r)
    out = ifftn(sig * res[None, ...], dim=(1, 2, 3))
    return torch.cat([out.real, out.imag], dim=0)

def perf_conv3d_speed(s, r, f):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    torch.cuda.synchronize()
    f(s, r)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)

def perf_conv3d_memory(s, r, f):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    f(s, r)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

def benchmark_conv3d_across_batches(batch_sizes, shape_s, shape_r, device='cuda'):
    times = {'rconv3d': [], 'cconv3d': [], 'split_conv3d': [], 'cconv3d_loop' : []}
    mems = {'rconv3d': [], 'cconv3d': [], 'split_conv3d': [], 'cconv3d_loop' : []}
    meminputs = []

    for nb in batch_sizes:
        print(f"Testing nbatches = {nb}")
        s = torch.rand((nb, *shape_s), device=device)
        r = torch.rand(shape_r, device=device)

        for name, fn in zip(['rconv3d', 'cconv3d', 'split_conv3d', 'cconv3d_loop'],
                            [rconv3d, cconv3d, split_conv3d, cconv3d_loop]):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            mem = perf_conv3d_memory(s, r, fn)
            perf_conv3d_speed(s, r, fn)
            time = perf_conv3d_speed(s, r, fn)


            times[name].append(time)
            mems[name].append(mem)

        o1 = cconv3d_loop(s, r)
        o2 = cconv3d(s, r)
        assert torch.allclose(o1, o2)

        input_bytes = (nb + 1) * np.prod([shape_s[i] + shape_r[i] - 1 for i in range(3)]) * s.element_size()
        meminputs.append(input_bytes / 1024**2)

    return times, mems, meminputs

def plot_perf_vs_batches(batch_sizes, times, mems, meminputs, shape_s, shape_r, figname):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, vals in times.items():
        plt.plot(batch_sizes, vals, 'o-', label=name)
    plt.xlabel('Number of Batches')
    plt.ylabel('Execution Time (ms)')
    plt.title('3D Convolution Runtime vs Batch Size')
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
    plt.title('3D Memory Usage vs Batch Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(figname)
    plt.show()


def test_conv3d():
    torch.manual_seed(10)
    torch.cuda.reset_peak_memory_stats()
    nbatches = 10
    ns = (10, 10, 10)
    nr = (5, 5, 5)
    s = torch.rand(ns).repeat(nbatches, 1, 1, 1).view(nbatches, *ns).to(torch.float64).to('cuda')
    r = torch.rand(nr).to(torch.float64).to('cuda')
    nsp = (ns[i] + nr[i]*2 - 2 for i in range(3))

    o = F.conv3d(F.pad(s, (4, 4, 4, 4, 4, 4), 'constant', 0).view(nbatches, 1, *nsp), r.flip(dims=(0,1,2)).view(1, 1, *nr), padding='valid')
    o = o.cpu()

    # test correctness
    o1 = rconv3d(s,r).cpu()
    o2 = cconv3d(s,r).cpu()
    o3 = split_conv3d(s,r).cpu()

    assert torch.allclose(o, o1)
    assert torch.allclose(o, o2)
    assert torch.allclose(o, o3)


def main():
    test_conv3d()

    for shape_s, shape_r in zip(
            [(20, 20, 20), (20, 20, 20), (20, 20, 20), (5, 5, 500), (50, 100, 100)],
            [(20, 20, 20), (1000, 20, 20), (20, 20, 1000), (9, 9, 6400), (50, 100, 100)]
    ):
        n = 1
        for i in range(3):
            n *= shape_s[i] + shape_r[i]
        if n > 3_000_000:
            batch_sizes = [10, 50, 100]
        elif n > 50_000:
            batch_sizes = [10, 50, 100, 200]
        else:
            batch_sizes = [10, 50, 100, 200, 400]

        size_str = f'sshape_{"_".join(map(str, shape_s))}_rshape_{"_".join(map(str, shape_r))}'

        times, mems, meminputs = benchmark_conv3d_across_batches(batch_sizes, shape_s, shape_r)
        plot_perf_vs_batches(batch_sizes, times, mems, meminputs, shape_s, shape_r, f'conv3d_resources_{size_str}.png')

        print(f"Finished plot conv3d_resources_{size_str}.png")

if __name__ == '__main__':
    main()
