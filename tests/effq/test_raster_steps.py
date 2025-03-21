import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import logging
import tred.graph as tg

logger = logging.getLogger('tred/tests/effq/test_raster_steps.py')

def test_compute_qeff_raster_steps(level=None):
    local_logger = logger.getChild('test_eval_qeff_raster_steps')
    if level:
        local_logger.setLevel(level)

    # Test the raster_steps wrapper. Note: raster_steps uses fixed npoints=(2,2,2)
    grid_spacing = (0.1, 0.1, 0.1)
    tail = torch.tensor([(0.4, 2.4, 3.4)])
    head = torch.tensor([(0.6, 2.6, 3.6)])
    sigma = torch.tensor([(0.5, 0.5, 0.5)])
    charge = torch.tensor([(1,)])
    nsigma = 0.7

    effq, offset = tg.raster_steps(grid_spacing, tail, head, sigma, charge, nsigma=nsigma)

    q_sum = torch.sum(effq).item()
    qint = 0.313723 # from mathematica
    local_logger.debug(f'Raster steps test passed, computed integral sum: {q_sum}')
    assert q_sum > 0, "Raster steps computed integral should be positive"
    assert np.isclose(q_sum, qint, rtol=1E-5), f'Raster steps computed integral does not match predefined value {qint}.'

def test_transform(level=None):
    local_logger = logger.getChild('test_transform')
    if level:
        local_logger.setLevel(level)
    local_logger.debug('Testing transform by predefined values')

    points = torch.tensor([3,4,5]).view(1,3)
    time = torch.tensor([2])
    pdims = (1, 2)
    tdim = 1
    raster = tg.Raster(velocity=0.16, grid_spacing=(1,1,1), pdims=pdims, tdim=tdim, nsigma=3.0)
    p = raster._transform(points, time)
    assert p.equal( torch.tensor([4, 2, 5]).view(1,3) ),  f'points after transformation is {p}.'

    points = torch.tensor([3,4,5]).view(1,3)
    time = torch.tensor([2])
    pdims = (0, 2)
    tdim = 1
    raster = tg.Raster(velocity=0.16, grid_spacing=(1,1,1), pdims=pdims, tdim=tdim, nsigma=3.0)
    p = raster._transform(points, time)
    assert p.equal(torch.tensor([3, 2, 5]).view(1,3)), f'points after transformation is {p}.'

    points = torch.tensor([3,4,5]).view(1,3)
    time = torch.tensor([2])
    pdims = (0, 2)
    tdim = -1
    raster = tg.Raster(velocity=0.16, grid_spacing=(1,1,1), pdims=pdims, tdim=tdim, nsigma=3.0)
    p = raster._transform(points, time)
    assert p.equal(torch.tensor([3, 5, 2]).view(1,3)), f'points after transformation is {p}.'

def test_time_diff():
    """Test the _time_diff method.
        Test _time_diff with head as None."""
    velocity = 2.0
    grid_spacing = torch.tensor([0.5])
    raster = tg.Raster(velocity, grid_spacing, pdims=())
    tail = torch.tensor([0.0, 1.0, 2.0])
    head = torch.tensor([2.0, 3.0, 4.0])
    result = raster._time_diff(tail, head)
    expected = (tail - head) / velocity
    assert torch.equal(result, expected), f"expected {expected}, result {result}"

    grid_spacing = torch.tensor([0.5, 0.5, 0.5])
    raster = tg.Raster(velocity, grid_spacing, pdims=(1,2))
    tail = torch.tensor([[0.0, 1.0, 2.0]])
    head = torch.tensor([[2.0, 3.0, 4.0]])
    result = raster._time_diff(tail, head)
    expected = (tail[:,0]-head[:,0]) / velocity
    assert torch.equal(result, expected)

    result = raster._time_diff(torch.tensor([1.0, 2.0, 3.0]))
    assert result is None


def test_drift_raster_positions():
    # Define parameters
    time = 1
    charge = torch.tensor([10, 10])# Placeholder value
    target = 3
    velocity = 2.0
    drifter = tg.Drifter(diffusion=[0.1, 0.1], lifetime=10, velocity=velocity, target=target)

    tail = torch.tensor([[1, 2], [-1, 1]])
    head = tail.clone().detach()
    head[:,0] = head[:,0] - 1
    head[:,1] = head[:,1] - 2
    _, dtime, _, dtail, dhead = drifter(time, charge, tail, head)

    # Create the plot
    # plt.figure(figsize=(8, 6))
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols, 6*nrows))

    # Plot Raster data points
    for i in range(tail.size(0)):
        if i == 0:
            axes[0].plot(tail[i,0], tail[i,1], color='blue', label='tail', marker='o', linestyle='-')
            axes[0].plot(head[i,0], head[i,1], color='red', label='head', marker='o', linestyle='-')
            axes[0].plot(dtail[i,0], dtail[i,1], color='blue', label='dtail', marker='x', linestyle='-')
            axes[0].plot(dhead[i,0], dhead[i,1], color='red', label='dhead', marker='x', linestyle='-')
        else:
            axes[0].plot(tail[i,0], tail[i,1], color='blue', marker='o', linestyle='-')
            axes[0].plot(head[i,0], head[i,1], color='red', marker='o', linestyle='-')
            axes[0].plot(dtail[i,0], dtail[i,1], color='blue', marker='x', linestyle='-')
            axes[0].plot(dhead[i,0], dhead[i,1], color='red', marker='x', linestyle='-')
        axes[0].plot([tail[i,0], head[i,0]], [tail[i, 1], head[i,1]], 'k--')
        axes[0].plot([dtail[i,0], dhead[i,0]], [dtail[i, 1], dhead[i,1]], 'k--')
    axes[0].vlines([target,], -2, 4, label='target')

    grid_spacing = torch.tensor([0.5, 0.5, 0.5])
    raster = tg.Raster(velocity, grid_spacing, pdims=(1,2))
    ttail = dtime
    thead = ttail + raster._time_diff(dtail, dhead)

    for i in range(tail.size(0)):
        if i == 0:
            axes[1].plot(time, tail[i,1], color='blue', label='tail', marker='o', linestyle='-')
            axes[1].plot(time, head[i,1], color='red', label='head', marker='o', linestyle='-')
            axes[1].plot(ttail[i], dtail[i,1], color='blue', label='dtail', marker='x', linestyle='-')
            axes[1].plot(thead[i], dhead[i,1], color='red', label='dhead', marker='x', linestyle='-')
        else:
            axes[1].plot(time, tail[i,1], color='blue', marker='o', linestyle='-')
            axes[1].plot(time, head[i,1], color='red', marker='o', linestyle='-')
            axes[1].plot(ttail[i], dtail[i,1], color='blue', marker='x', linestyle='-')
            axes[1].plot(thead[i], dhead[i,1], color='red', marker='x', linestyle='-')
        axes[1].plot([time, time], [tail[i, 1], head[i,1]], 'k--')
        axes[1].plot([ttail[i], thead[i]], [dtail[i, 1], dhead[i,1]], 'k--')
    axes[1].vlines([time,], -2, 4, label='initial time')

    # Labels and legend
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    axes[0].set_title("Tail, Head, before and after drifting")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Y-axis")
    axes[1].set_title("Tail, Head, before and after driting")
    axes[1].legend()
    axes[1].grid(True)

    # Show the plot
    plt.savefig('drifted_tailhead.png')

def test_raster(level=None):
    local_logger = logger.getChild('test_raster')
    if level:
        local_logger.setLevel(level)

    # Test the raster_steps wrapper. Note: raster_steps uses fixed npoints=(2,2,2)
    velocity = 0.1
    target = 0.1
    grid_spacing = (0.1, 0.1, 0.1) # 0.1 * velocity = 0.01, tdim = -1

    sigma = torch.tensor([(0.05, 0.5, 0.5)]) # 0.5 * velocity = 0.05
    charge = torch.tensor([(1,)])
    nsigma = 0.7

    tail = torch.tensor([(0.06, 2.4, 3.4)]) # 0.06 to 0.1 = 0.4 in time
    head = torch.tensor([(0.04, 2.6, 3.6)]) # 0.04 to 0.1 = 0.6 in time

    drifter = tg.Drifter(diffusion=[0.1, 0.1, 0.1], lifetime=10, velocity=velocity, target=target)

    _, dtime, _, dtail, dhead = drifter(None, charge, tail, head)

    raster = tg.Raster(velocity, grid_spacing, pdims=(1,2), tdim=-1, nsigma=nsigma)
    block = raster(sigma, dtime, charge, dtail, dhead)

    q_sum = torch.sum(block.data).item()
    qint = 0.313723 # from mathematica
    local_logger.debug(f'Raster steps test passed, computed integral sum: {q_sum}')
    assert q_sum > 0, "Raster steps computed integral should be positive"
    assert np.isclose(q_sum, qint, rtol=1E-5), f'Raster steps computed integral, {q_sum}, does not match predefined value {qint}.'

def main():
    # print('-------- test_compute_qeff ---------')
    test_compute_qeff_raster_steps()

    # print('-------- test_transform ---------')
    test_transform()

    test_time_diff()

    test_drift_raster_positions()

    test_raster()

if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    try:
        opt = sys.argv[1]
        if opt.lower() == 'debug':
            logger.setLevel(level=logging.DEBUG)
        elif opt.lower() == 'warning':
            logger.setLevel(level=logging.WARNING)
        elif opt.lower() == 'info':
            logger.setLevel(level=logging.INFO)
        else:
            print('Usage: test_grid.py [debug|warning|info]')
            exit(-1)
    except IndexError:
        # logging.basicConfig(level=logging.DEBUG)
        print('To use system default logging level')

    main()
