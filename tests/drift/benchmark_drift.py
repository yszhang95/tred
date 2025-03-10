#!/usr/bin/env python

import torch
import torch.utils.benchmark as benchmark
import matplotlib.pyplot as plt
import sys

from tred.graph import Drifter

import numpy as np

#############################################
# Benchmark script using torch.utils.benchmark
# and basic memory usage checks.
#############################################

def run_benchmark(
    npt: int,
    device: str,
    repeats: int = 100,
):
    """
    Construct random input and measure execution time and approximate
    peak memory usage on the given device.
    """

    # Prepare random input data
    # shapes: time, charge: (npt,)
    # tail, head: (npt, 3)
    time_data = torch.zeros(npt, dtype=torch.float32)  # some or random times
    charge_data = torch.randint(800, 1200, (npt,), dtype=torch.int32).to(torch.float32)
    tail_data = torch.randn(npt, 3, dtype=torch.float32)
    head_data = torch.randn(npt, 3, dtype=torch.float32)

    # Send to device
    time_data = time_data.to(device)
    charge_data = charge_data.to(device)
    tail_data = tail_data.to(device)
    head_data = head_data.to(device)

    # Create Drifter object
    drifter = Drifter(
        diffusion=[5.0, 5.0, 5.0],
        lifetime=2.0,
        velocity=1.0,
        target=0.0,
        vaxis=0,
        fluctuate=False,
        drtoa=1.
    ).to(device)

    # We create a simple callable for the benchmark
    def drifter_forward():
        with torch.no_grad():
            drifter(time_data, charge_data, tail_data, head_data)

    # --- Time measurement ---
    t = benchmark.Timer(
        stmt='drifter_forward()',
        globals={'drifter_forward': drifter_forward},
    )#.timeit(repeats)
    m = t.blocked_autorange()

    # We'll capture the average time per run
    avg_time_ms = m.mean * 1E3 #ms

    # --- Memory measurement ---
    #
    # For GPU:
    #   Reset peak stats, run once more, then measure.
    # For CPU:
    #   We approximate memory usage by summing the sizes of the input and output.
    #   A more sophisticated approach may be used if needed.
    peak_mem_bytes = 0
    input_mem_bytes = 0
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            drifter(time_data, charge_data, tail_data, head_data)
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        # Input memory
        input_mem_bytes += time_data.element_size() * time_data.nelement()
        input_mem_bytes += charge_data.element_size() * charge_data.nelement()
        input_mem_bytes += tail_data.element_size() * tail_data.nelement()
        input_mem_bytes += head_data.element_size() * head_data.nelement()
    else:
        # CPU "peak" is trickier to measure precisely without external libraries.
        # We'll do a naive sum of input sizes as "peak" for demonstration.
        # If you want real usage, consider memory_profiler or psutil.
        with torch.no_grad():
            drifter(time_data, charge_data, tail_data, head_data)

        # Just approximate everything for demonstration:
        time_mem = time_data.element_size() * time_data.nelement()
        charge_mem = charge_data.element_size() * charge_data.nelement()
        tail_mem = tail_data.element_size() * tail_data.nelement()
        head_mem = head_data.element_size() * head_data.nelement()

        # We'll say the model also has some overhead:
        model_mem = sum(p.element_size() * p.nelement() for p in drifter.parameters())

        peak_mem_bytes = time_mem + charge_mem + tail_mem + head_mem + model_mem
        input_mem_bytes = time_mem + charge_mem + tail_mem + head_mem

    # We measure the ratio of peak memory usage to the size of just the input tensors:
    mem_ratio = float(peak_mem_bytes) / float(input_mem_bytes) if input_mem_bytes else 0.0

    return avg_time_ms, mem_ratio, peak_mem_bytes, input_mem_bytes


def main():
    # Vary npt (number of points)
    npt_list = [1000, 5000, 10000, 50000, 100000, 1_000_000, 5_000_000, 10_000_000]

    cpu_results_time = []
    cpu_results_memratio = []
    gpu_results_time = []
    gpu_results_memratio = []
    gpu_results_peak_mem = []
    gpu_results_mem = []

    # Run on CPU
    for npt in npt_list:
        t, _, _, _ = run_benchmark(npt, 'cpu', repeats=50)
        cpu_results_time.append(t)
        print(f"CPU npt={npt}: time={t:.3f} ms")

    # Check if CUDA is available, then run on GPU
    if torch.cuda.is_available():
        for npt in npt_list:
            t, r, pm, m = run_benchmark(npt, 'cuda', repeats=50)
            gpu_results_time.append(t)
            gpu_results_memratio.append(r)
            gpu_results_peak_mem.append(pm)
            gpu_results_mem.append(m)
            print(f"GPU npt={npt}: time={t:.3f} ms")
    else:
        print("CUDA is not available. GPU benchmarks skipped.")

    # --- Plot the results ---
    # 1) Time vs. npt
    plt.figure()
    plt.plot(npt_list, cpu_results_time, 'o-', label='CPU')
    if gpu_results_time:
        plt.plot(npt_list, gpu_results_time, 'ro-', label='GPU')
    for x, y in zip(npt_list, gpu_results_time):
        plt.text(x, y, f'{y:.2f} ms', fontsize=10, ha='right', va='bottom', color='r')
    plt.xlabel('Input length (npt)')
    plt.xscale('log')
    plt.ylabel('Average execution time (ms)')
    plt.title('Drifter Benchmark: Execution Time vs Input Length')
    plt.legend()
    plt.show()
    plt.savefig('benchmark_running_time.png')

    plt.figure()
    plt.plot(gpu_results_mem, cpu_results_time, 'o-', label='CPU')
    if gpu_results_time:
        plt.plot(gpu_results_mem, gpu_results_time, 'ro-', label='GPU')
    for x, y in zip(gpu_results_mem, gpu_results_time):
        plt.text(x, y, f'{y:.2f} ms', fontsize=10, ha='right', va='bottom', color='r')
    # Annotate each GPU point with its value
    plt.xlabel('Mem for input data [MB]')
    plt.xscale('log')
    plt.ylabel('Average execution time (ms)')
    plt.title('Drifter Benchmark: Execution Time vs Input Length')
    plt.legend()
    plt.show()
    plt.savefig('benchmark_running_time_mem.png')

    # 2) Memory ratio vs. npt
    plt.figure()
    if gpu_results_time:
        plt.plot(npt_list, gpu_results_memratio, 'o-', label='GPU')
    plt.xlabel('Input length (npt)')
    plt.ylabel('Peak Mem / Input Mem')
    plt.title('Drifter Benchmark: Memory Ratio vs Input Length')
    plt.legend()
    plt.show()

    plt.savefig('benchmark_mem_ratio.png')

    # 3) Memory vs. npt
    plt.figure()
    if gpu_results_time:
        plt.plot(npt_list, np.array(gpu_results_mem)/1024**2, 'o-', label='GPU')
    plt.xlabel('Input length (npt)')
    plt.xscale('log')
    plt.ylabel('Peak Mem (MB)')
    plt.title('Drifter Benchmark: Peak memory vs Input Length')
    plt.legend()
    plt.show()

    plt.savefig('benchmark_peak_mem.png')

if __name__ == "__main__":
    main()
