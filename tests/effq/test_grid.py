import tred.raster.steps as ts
from tred.raster.steps import compute_index, compute_coordinate

from tred.types import index_dtype, MAX_INDEX, MIN_INDEX

import torch

def test_grid():
    # Define grid parameters
    origin = (0.0, 0.0, 0.0)
    grid_spacing = (1.0, 1.0, 1.0)
    print('setup')
    print('origin', origin)
    print('spacing', grid_spacing)

    test_coords = [[3.5, 3., 4.], [1.1, 2.2, 3.3]]
    predefined_indices = torch.tensor([[3, 3, 4], [1, 2, 3]],
                                      requires_grad=False, dtype=index_dtype)
    indices = compute_index(test_coords, origin, grid_spacing)
    assert torch.equal(indices, predefined_indices), f'Not equal to f{predefined_indices}'
    print(f'Indices for {test_coords} ->\n', indices)
    indices = compute_index(test_coords[0], origin, grid_spacing)
    print(f'Indices for {test_coords[0]} ->\n', indices)

    test_indices = [(2,3,4),(3,4,5)]
    predefined_coords = torch.tensor(
        [[2.,3.,4.], [3.,4.,5.]],
        requires_grad=False, dtype=torch.float32
    )
    coords = compute_coordinate(test_indices, origin, grid_spacing)
    print(f"Coordinates for indices f{test_indices} ->\n", coords)
    assert torch.allclose(coords, predefined_coords, atol=1E-6, rtol=1E-6), \
        f'Not equal to f{predefined_coords}'
    coords = compute_coordinate(test_indices[0], origin, grid_spacing)
    print(f"Coordinates for indices f{test_indices[0]} ->\n", coords)

def main():
    test_grid()

if __name__ == '__main__':
    print(MAX_INDEX, MIN_INDEX)
    main()
