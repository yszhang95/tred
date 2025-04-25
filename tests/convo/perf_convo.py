import torch.profiler

import tred.convo
from tred.blocking import Block
from tred.partitioning import deinterlace_pairs
from tred.response import ndlarsim
from tred.convo import dft_shape, signal_pad, response_pad, zero_pad

import time

# uncoment the lines marked with zero padding to test zeros padding
# uncoment the lines marked with symemtric padding to test symmetric padding
# uncoment the lines marked with built-in padding to test built-in pading

def convolve_spec(signal: Block, response_spec, taxis: int = -1) -> Block:
    iscomplex = False
    if signal.data.dtype == torch.complex64 or signal.data.dtype == torch.complex128:
        iscomplex = True

    # with torch.profiler.record_function("signal_pad"): # zero padding / symmetric padding
    with torch.profiler.record_function("signal_shift"): # built-in padding
        nrm1 = [i - j for i,j in zip(response_spec.shape, signal.data.size()[1:])] # Nr - 1 # built-in padding
        nrm1[taxis] = 0 # built-in padding
        if any(x % 2 != 0 for x in nrm1): # built-in padding
            raise ValueError(f"Length of response tensor must always be odd. {nrm1} + 1 is given.") # built-in padding
        nrm1 = torch.tensor(nrm1, device=signal.data.device) # built-in padding
        fh = nrm1 // 2 # built-in padding
        signal = Block(signal.location - fh, data=signal.data) # built-in padding
        # signal = signal_pad(signal, response_spec.shape, taxis) # symmetric padding
        # signal.data = zero_pad(signal.data, response_spec.shape) # zero padding

    # exclude first batched dimension
    dims = (torch.arange(signal.vdim) + 1).tolist()

    with torch.profiler.record_function("signal_fft"):
        spec = torch.fft.fftn(signal.data, s=response_spec.shape, dim=dims)
        signal.data = None # manual release memory
    with torch.profiler.record_function("signal_response_mult"):
        measure = spec * response_spec
    with torch.profiler.record_function("ifft"):
        measure = torch.fft.ifftn(measure, dim=dims)
    if not iscomplex:
        measure = measure.real
    return Block(location = signal.location, data = measure) # fixme: normalization

def convolve(signal, response, taxis: int = -1) -> Block:
    with torch.profiler.record_function("convolve_prep"):
        c_shape = dft_shape(signal.shape, response.shape)
        dims = torch.arange(len(c_shape)).tolist()
    # with torch.profiler.record_function("response_pad"): # zero padding / symmetric padding
    #     response = response_pad(response, c_shape, taxis) # symmetric padding
    #     # response = zero_pad(response, c_shape) # zero padding
    with torch.profiler.record_function("response_fft"):
        response_spec = torch.fft.fftn(response, s=c_shape, dim=dims)
    return convolve_spec(signal, response_spec, taxis)

def interlaced_symm(signal, response, steps, symm_axis=0):
    with torch.profiler.record_function("interlaced_symm"):
        super_location = signal.location // steps
        batched_steps = torch.cat([torch.tensor([1], device=steps.device), steps])
        sig_laces = deinterlace_pairs(signal.data, batched_steps, 1+symm_axis) # one extra dim for batch dim
        res_laces = deinterlace_pairs(response, steps, symm_axis)

        flipdims = (1+symm_axis,) # batch dim == 0

        meas = None
        for sig_lace, res_lace in zip(sig_laces, res_laces):
            sig_lace_block = Block(super_location, data=torch.complex(sig_lace[0], sig_lace[1].flip(dims=flipdims)))
            # res_lace[1] is not used as it is a flipped copy, given the reflection symmetry
            meas_lace_block = convolve(sig_lace_block, res_lace[0])
            meas_lace_block = Block(location = meas_lace_block.location,
                                data = meas_lace_block.data.real + torch.flip(meas_lace_block.data.imag, dims=flipdims))
            if meas is None:
                meas = meas_lace_block
                continue
            meas.data += meas_lace_block.data
        return meas


def perf_interlaced_symm():
    response = ndlarsim('response_38_v2b_50ns_ndlar.npy')
    response = response.to('cuda')
    data = torch.rand((50, 80,80, 513), device='cuda')
    location = torch.ones((data.size(0), 3), dtype=torch.int32, device='cuda')
    signal = Block(data=data, location=location)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False
    ) as prof:

        interlaced_symm(signal, response, torch.tensor((10,10,1), device='cuda'))

    entries = prof.key_averages()
    # cuda events only
    entries = [e for e in prof.key_averages()
               if e.device_time_total > 0 and e.cpu_time_total == 0]

    # Filter out raw CUDA kernels (e.g., cuFFT kernels like void multi_bluestein_fft...)
    filtered = [e for e in entries if not e.key.startswith("void")]

    # Optional: also filter out internal PyTorch ops
    filtered = [e for e in filtered if not e.key.startswith("aten::")]
    sorted_filtered = sorted(filtered, key=lambda e: e.device_time_total, reverse=True)

    print('Summary table from profiler')
    for e in sorted_filtered[:10]:  # top 10
        print(f"{e.key:<40} |CUDA total: {e.device_time_total / 1E3:11.5f}ms | "
              f"Self CUDA: {e.self_device_time_total / 1E3:11.5f}ms | Calls: {e.count}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history()
    tred.convo.interlaced_symm(signal, response, torch.tensor((10,10,1), device='cuda'))
    torch.cuda.memory._dump_snapshot("convolution_interlaced_sym.pickle")
    print(f"Peak CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2} MB")
    torch.cuda.synchronize()
    s = time.time()
    tred.convo.interlaced_symm(signal, response, torch.tensor((10,10,1), device='cuda'))
    torch.cuda.synchronize()
    e = time.time()
    print('Manual Measured time', (e-s)*1E3, 'ms')
    print('Finished analyzing interlaced_symm\n')

def perf_interlaced_symm_v2():
    response = ndlarsim('response_38_v2b_50ns_ndlar.npy')
    response = response.to('cuda')
    data = torch.rand((50, 80,80, 513), device='cuda')
    location = torch.ones((data.size(0), 3), dtype=torch.int32, device='cuda')
    signal = Block(data=data, location=location)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False
    ) as prof:

        tred.convo.interlaced_symm_v2(signal, response, torch.tensor((10,10,1), device='cuda'))

    entries = prof.key_averages()
    # cuda events only
    entries = [e for e in prof.key_averages()
               if e.device_time_total > 0]

    filtered = [e for e in entries if e.key.startswith("aten::")]
    sorted_filtered = sorted(filtered, key=lambda e: e.device_time_total, reverse=True)
    # print(sorted_filtered[0], sorted_filtered[0].self_device_time_total)

    print('Summary table from profiler')
    for e in sorted_filtered[:10]:  # top 10
        print(f"{e.key:<40} |CUDA total: {e.device_time_total / 1E3 :11.5f}ms | "
              f"Self CUDA: {e.self_device_time_total / 1E3 :11.5f}ms | Calls: {e.count}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history()
    tred.convo.interlaced_symm_v2(signal, response, torch.tensor((10,10,1), device='cuda'))
    torch.cuda.memory._dump_snapshot("convolution_interlaced_symm_v2.pickle")
    print(f"Peak CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2} MB")

    torch.cuda.synchronize()
    s = time.time()
    tred.convo.interlaced_symm_v2(signal, response, torch.tensor((10,10,1), device='cuda'))
    torch.cuda.synchronize()
    e = time.time()
    print('Manual Measured time', (e-s)*1E3, 'ms')
    print('Finished analyzing interlaced_symm_v2\n')

def main():
    perf_interlaced_symm()
    perf_interlaced_symm_v2()

if __name__ == '__main__':
    main()
