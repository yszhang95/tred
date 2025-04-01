from tred.indexing import crop_batched

import torch
import torch.utils.benchmark as benchmark
import time

def benchmark_crop_batched():
    # prepare for test data
    device = 'cuda'
    torch.random.manual_seed(10)
    inner = torch.tensor([30,30,30], device=device)
    outer = torch.tensor([100,100,100], device=device)
    offsets = torch.randint(10, 60, (10_000, 3), device=device)

    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )

    # Run your PyTorch Model.
    # At any point in time, save a snapshot to file for later.
    with torch.no_grad():
        crop_batched(offsets, inner, outer)
    # In this sample, we save the snapshot after running code above
    #   - Save as many snapshots as you'd like.
    #   - Snapshots will save last `max_entries` number of memory events
    #     (100,000 in this example).
    try:
        torch.cuda.memory._dump_snapshot(f"crop_batched.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")

    # Stop recording memory snapshot history.
    torch.cuda.memory._record_memory_history(enabled=None)


    # record time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    # Code to benchmark
    with torch.no_grad():
        inds = crop_batched(offsets, inner, outer)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'Elpased {elapsed_time_ms} ms on GPU')
    print(f'Resulting indices in a shape of {inds.shape}, on device {inds.device}')


    start_time = time.time()
    # Code to benchmark
    with torch.no_grad():
        offsets = offsets.to('cpu')
        inner = inner.to('cpu')
        outer = outer.to('cpu')
        inds = crop_batched(offsets, inner, outer)
    end_time = time.time()
    elapsed_time_ms = 1E3*(end_time - start_time)
    print(f'Elpased {elapsed_time_ms} ms on CPU')
    print(f'Resulting indices in a shape of {inds.shape}, on device {inds.device}')

def main():
    benchmark_crop_batched()


if __name__ == '__main__':
    main()
