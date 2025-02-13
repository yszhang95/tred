import tred.raster.steps as ts
from tred.raster.steps import (
    compute_index, compute_coordinate,
    _stack_X0X1, compute_charge_box,
    compute_bounds_X0_X1, compute_bounds_X0X1,
    reduce_to_universal
)


from tred.types import index_dtype, MAX_INDEX, MIN_INDEX

import logging
import sys
import torch

logger = logging.getLogger('tred/tests/effq/test_grid.py')

def test_grid():
    local_logger = logger.getChild('test_grid')
    # Define grid parameters
    origin = (0.0, 0.0, 0.0)
    grid_spacing = (1.0, 1.0, 1.0)

    local_logger.debug(f'Setup: origin {origin}')
    local_logger.debug(f'Setupg spacing {grid_spacing}')

    test_coords = [[3.5, 3., 4.], [1.1, 2.2, 3.3]]
    predefined_indices = torch.tensor([[3, 3, 4], [1, 2, 3]],
                                      requires_grad=False, dtype=index_dtype)
    indices = compute_index(torch.tensor(test_coords), origin, grid_spacing)
    assert torch.equal(indices, predefined_indices), f'Not equal to f{predefined_indices}'
    local_logger.debug(f'Indices for {test_coords} ->\n {indices}')
    indices = compute_index(torch.tensor(test_coords[0]), origin, grid_spacing)
    local_logger.debug(f'Indices for {test_coords[0]} ->\n {indices}')

    test_indices = [(2,3,4),(3,4,5)]
    predefined_coords = torch.tensor(
        [[2.,3.,4.], [3.,4.,5.]],
        requires_grad=False, dtype=torch.float32
    )
    coords = compute_coordinate(torch.tensor(test_indices), origin, grid_spacing)
    local_logger.debug(f"Coordinates for indices f{test_indices} ->\n {coords}")
    assert torch.allclose(coords, predefined_coords, atol=1E-6, rtol=1E-6), \
        f'Not equal to f{predefined_coords}'
    coords = compute_coordinate(torch.tensor(test_indices[0]), origin, grid_spacing)
    local_logger.debug(f"Coordinates for indices f{test_indices[0]} ->\n {coords}")

def test_compute_charge_box():
    local_logger = logger.getChild('test_compute_charge_box')
    # Initial setup
    origin=(-1,-1,-1)
    grid_spacing=(1,1,1)
    n_sigma=(5,3,1)

    local_logger.debug(f'Initial setup: origin {origin}')
    local_logger.debug(f'Initial setup: grid_spacing {grid_spacing}')
    local_logger.debug(f'Initial setup: n_sigma {n_sigma}')

    # Input data
    X0 = torch.tensor([[1.0, 1.0, 1.0], [6, 2.0, 2.0]],
                      dtype=torch.float32)  # Starting points
    X1 = torch.tensor([[3.0, 3.0, 3.0], [5.0, 5.0, 5.0]],
                      dtype=torch.float32)  # Ending points
    Sigma = torch.tensor([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]],
                         dtype=torch.float32)  # Diffusion widths

    local_logger.debug('\nInput X0 {}\nX1 {}\nSigma {}'.format(X0, X1, Sigma))

    # test _stack_X0X1

    local_logger.debug('_stack_X0X1')
    X0X1 = _stack_X0X1(X0, X1)
    for i in range(X0.shape[0]):
        for j in range(X0.shape[1]):
            assert torch.allclose(X0X1[i,j][0], X0[i,j]) \
                and torch.allclose(X0X1[i,j][1], X1[i,j]), \
                f'_stack_X0X1 failed at X0[{i},{j}], X1[{i},{j}], X0X1[{i},{j}]'
            local_logger.debug(f'X0X1[{j},0]:{X0X1[i,j][0]}, X0[{j}]:{X0[i,j]}, '
                               f'X0X1[{j},1]:{X0X1[i,j][1]}, X1[{j}]:{X1[i,j]}')

    local_logger.debug('reduce_to_universal')
    test_shapes = torch.tensor([[1,2], [3, 4], [4, 1]], dtype=torch.int32)
    universal_max = reduce_to_universal(test_shapes)
    local_logger.debug(f'Test input {test_shapes}')
    local_logger.debug(f'Test output {universal_max}')
    for i in range(len(test_shapes)):
        assert torch.all(universal_max >= test_shapes[i]), \
            f'Index {i}, max: {universal_max}, test_shapes[{i}] {test_shapes[i]}'

    # Test compute_bounds_X0_X1; no recentering
    local_logger.debug('compute_bounds_X0X1')
    bounds = compute_bounds_X0X1(X0X1, Sigma, n_sigma)
    for i in range(X0.shape[0]):
        for j in range(X0.shape[1]):
            # check bounds follows in between X0,X1,+/-Sigma*nsigma
            x0 = min(X0X1[i,j,0], X0X1[i,j,1])
            x1 = max(X0X1[i,j,0], X0X1[i,j,1])
            exact_bounds = (x0 - n_sigma[j] * Sigma[i,j],
                            x1 + n_sigma[j] * Sigma[i,j])
            msg = f'Index, ({i},{j}), lower bound {bounds[i,j,0]}, '\
                f'upper bound {bounds[i,j,1]}, ' \
                f'X0 {X0X1[i,j,0]}, X1 {X0X1[i,j,1]}, '\
                f'n_sigma {n_sigma[j]}, Sigma {Sigma[i,j]}, '\
                f'lower X0X1: {x0}, upper X0X1: {x1}, '\
                f'exact lower bound: {exact_bounds[0]}, exact upper bound: {exact_bounds[1]}'
            local_logger.debug(msg)
            assert bounds[i,j,0] <= exact_bounds[0] and \
                        bounds[i,j,1] >= exact_bounds[1], \
                        f'compute_bounds_X0X1 failed at X0={X0X1[i,j,0]}, '\
                        f'X1={X0X1[i,j,1]}, Sigma={Sigma[i,j]}, , n_sigma={n_sigma[j]}, '\
                        f'exact lower bound: {exact_bounds[0]}, exact upper bound: {exact_bounds[1]}, '\
                        f'lower bound: {bounds[i,j,0]}, upper bound: {bounds[i,j,1]}'

    local_logger.debug('compute_bounds_X0_X1')
    bounds = compute_bounds_X0_X1(X0, X1, Sigma, n_sigma)
    for i in range(X0.shape[0]):
        for j in range(X0.shape[1]):
            # check bounds follows in between X0,X1,+/-Sigma*nsigma
            x0 = min(X0[i,j], X1[i,j])
            x1 = max(X0[i,j], X1[i,j])
            exact_bounds = (x0 - n_sigma[j] * Sigma[i,j],
                            x1 + n_sigma[j] * Sigma[i,j])
            msg = f'Index, ({i},{j}), lower bound {bounds[i,j,0]}, '\
                f'upper bound {bounds[i,j,1]}, ' \
                f'X0 {X0[i,j]}, X1 {X1[i,j]}, '\
                f'n_sigma {n_sigma[j]}, Sigma {Sigma[i,j]},'\
                f'lower X0X1: {x0}, upper X0X1: {x1}, '\
                f'exact lower bound: {exact_bounds[0]}, exact upper bound: {exact_bounds[1]}'
            local_logger.debug(msg)
            assert bounds[i,j,0] <= exact_bounds[0] and \
                        bounds[i,j,1] >= exact_bounds[1], \
                        f'compute_bounds_X0_X1 failed at X0={X0[i,j]}, '\
                        f'X1={X1[i,j]}, Sigma={Sigma[i,j]}, , n_sigma={n_sigma[j]}, '\
                        f'exact lower bound: {exact_bounds[0]}, exact upper bound: {exact_bounds[1]}, '\
                        f'lower bound: {bounds[i,j,0]}, upper bound: {bounds[i,j,1]}'

    # Compute charge box
    local_logger.debug('compute_charge_box, without recentering')
    result = compute_charge_box(X0, X1, Sigma, n_sigma, origin, grid_spacing)
    idx_bounds_up = result[0] + result[1].unsqueeze(0)
    bounds_lw = compute_coordinate(result[0], origin, grid_spacing)
    bounds_up = compute_coordinate(idx_bounds_up, origin, grid_spacing)
    for i in range(X0.size(0)):
        for j in range(X0.size(1)):
            lower = min(X0[i,j], X1[i,j])
            upper = max(X0[i,j], X1[i,j])
            x0 = lower
            x1 = upper
            exact_bounds = (x0 - n_sigma[j] * Sigma[i,j],
                            x1 + n_sigma[j] * Sigma[i,j])
            msg = f'Index, ({i},{j}), lower bound {bounds_lw[i,j]}, '\
                f'upper bound {bounds_up[i,j]}, ' \
                f'X0 {X0[i,j]}, X1 {X1[i,j]}, '\
                f'n_sigma {n_sigma[j]}, Sigma {Sigma[i,j]},'\
                f'lower X0X1: {x0}, upper X0X1: {x1}, '\
                f'exact lower bound: {exact_bounds[0]}, exact upper bound: {exact_bounds[1]}'
            local_logger.debug(msg)
            assert (bounds_lw[i,j] <= exact_bounds[0] and
                        bounds_up[i,j] >= exact_bounds[1]), \
                        f'compute_charge_box without recentering failed at X0={X0[i,j]}, '\
                        f'X1={X1[i,j]}, Sigma={Sigma[i,j]}, , n_sigma={n_sigma[j]}, '\
                        f'exact lower bound: {exact_bounds[0]}, exact upper bound: {exact_bounds[1]}, '\
                        f'lower bound={bounds_lw[i,j]}, upper bound={bounds_up[i,j]}'


    # Compute charge box
    local_logger.debug('compute_charge_box, with recentering')
    result = compute_charge_box(X0, X1, Sigma, n_sigma, origin, grid_spacing, recenter=True)
    idx_bounds_up = result[0] + result[1].unsqueeze(0)
    bounds_lw = compute_coordinate(result[0], origin, grid_spacing)
    bounds_up = compute_coordinate(idx_bounds_up, origin, grid_spacing)
    for i in range(X0.size(0)):
        for j in range(X0.size(1)):
            lower = min(X0[i,j], X1[i,j])
            upper = max(X0[i,j], X1[i,j])
            x0 = lower
            x1 = upper
            exact_bounds = (x0 - n_sigma[j] * Sigma[i,j],
                            x1 + n_sigma[j] * Sigma[i,j])
            msg = f'Index ({i}, {j}), lower bound {bounds_lw[i,j]}, '\
                f'upper bound {bounds_up[i,j]}, ' \
                f'X0 {X0[i,j]}, X1 {X1[i,j]}, '\
                f'n_sigma {n_sigma[j]}, Sigma {Sigma[i,j]}, '\
                f'lower X0X1: {x0}, upper X0X1: {x1}, '\
                f'exact lower bound: {exact_bounds[0]}, exact upper bound: {exact_bounds[1]}'
            local_logger.debug(msg)
            assert (bounds_lw[i,j] <= exact_bounds[0] and
                        bounds_up[i,j] >= exact_bounds[1]), \
                        f'compute_charge_box with recentering failed at X0={X0[i,j]}, '\
                        f'X1={X1[i,j]}, Sigma={Sigma[i,j]}, , n_sigma={n_sigma[j]}, '\
                        f'exact lower bound: {exact_bounds[0]}, exact upper bound: {exact_bounds[1]}, '\
                        f'lower bound={bounds_lw[i,j]}, upper bound={bounds_up[i,j]}'

def main():
    print('----- test_grid() ------')
    test_grid()

    print('----- test_compute_charge_box() ------')
    test_compute_charge_box()

if __name__ == '__main__':

    try:
        opt = sys.argv[1]
        if opt.lower() == 'debug':
            logging.basicConfig(level=logging.DEBUG)
        elif opt.lower() == 'warning':
            logging.basicConfig(level=logging.WARNIGN)
        elif opt.lower() == 'info':
            logging.basicConfig(level=logging.INFO)
        else:
            print('Usage: test_grid.py [debug|warning|info]')
            exit(-1)
    except IndexError:
        # logging.basicConfig(level=logging.DEBUG)
        print('To use system default logging level')
    main()
