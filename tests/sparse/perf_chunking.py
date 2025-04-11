from tred.chunking import accumulate
from tred.blocking import Block

import torch
import torch.utils.benchmark as benchmark
import time

from torch.distributions import Poisson, Exponential

def benchmark_accumulate():
    # prepare for test data
    device = 'cuda'
    torch.random.manual_seed(10)
    # occurrence= Poisson(torch.tensor(3.)).sample((2000,)).to(torch.int32)
    occurrence= Exponential(torch.tensor(0.1)).sample((2000,)).to(torch.int32).clamp(max=50) + 1
    nevents = torch.sum(occurrence).item()
    noccur = torch.unique(occurrence)
    offsets = torch.zeros((nevents, 3), dtype=torch.int32, device=device)

    uniq_offsets = torch.randint(10, 60, (len(noccur), 3), device=device)
    j = 0
    for i, n in enumerate(noccur):
        offsets[j:j+n] = uniq_offsets[i]
        j = j+n
    offsets = offsets[torch.randperm(offsets.size(0))]

    data = torch.ones((len(offsets), 50,50,50), device=device)

    removed = torch.randint(0, len(offsets), (len(offsets)//2,))
    data[removed] = 0

    print('Number of unique offsets:', len(torch.unique(offsets,dim=0)))
    print('Number of offsets:', len(offsets))

    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )

    # Run your PyTorch Model.
    # At any point in time, save a snapshot to file for later.
    with torch.no_grad():
        accumulate(Block(location=offsets, data=data))
    # In this sample, we save the snapshot after running code above
    #   - Save as many snapshots as you'd like.
    #   - Snapshots will save last `max_entries` number of memory events
    #     (100,000 in this example).
    try:
        torch.cuda.memory._dump_snapshot(f"accumulate.pickle")
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
        accumulate(Block(location=offsets, data=data))
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'Elpased {elapsed_time_ms} ms on GPU')


    start_time = time.time()
    # Code to benchmark
    with torch.no_grad():
        offsets = offsets.to('cpu')
        data = data.to('cpu')
        accumulate(Block(location=offsets, data=data))
    end_time = time.time()
    elapsed_time_ms = 1E3*(end_time - start_time)
    print(f'Elpased {elapsed_time_ms} ms on CPU')

def main():
    benchmark_accumulate()


if __name__ == '__main__':
    main()
