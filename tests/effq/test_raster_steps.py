import sys
import numpy as np

import torch
import logging
import tred.graph as tg

def test_compute_qeff_raster_steps(level=None):
    local_logger = logging.getLogger('test_eval_qeff_raster_steps')
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

def main():
    print('-------- test_compute_qeff ---------')
    test_compute_qeff_raster_steps()


if __name__ == '__main__':
    try:
        opt = sys.argv[1]
        if opt.lower() == 'debug':
            logging.basicConfig(level=logging.DEBUG)
        elif opt.lower() == 'warning':
            logging.basicConfig(level=logging.WARNING)
        elif opt.lower() == 'info':
            logging.basicConfig(level=logging.INFO)
        else:
            print('Usage: test_grid.py [debug|warning|info]')
            exit(-1)
    except IndexError:
        # logging.basicConfig(level=logging.DEBUG)
        print('To use system default logging level')

    main()
