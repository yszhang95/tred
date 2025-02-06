import tred.raster.steps as ts
from tred.raster.steps import (
    compute_index, compute_coordinate, compute_charge_box,
    qline_diff3D,
)

import json
import logging
import sys

import torch

logger = logging.getLogger('tred/tests/effq/test_effq.py')

def test_QModel():
    local_logger = logger.getChild('test_QModel')
    local_logger.debug('Testing qmodel by comparing results with Mathematica')
    Q = torch.tensor([1]).view(1)
    X0 = torch.tensor((0.4,2.4,3.4), dtype=torch.float64).view(1, 3)
    X1 = torch.tensor((0.6, 2.6, 3.6), dtype=torch.float64).view(1, 3)
    Sigma = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64).view(1,3)

    x = torch.linspace(0.2, 0.8, 5, dtype=torch.float64).view(1,-1)
    y = torch.linspace(2.2, 2.8, 5, dtype=torch.float64).view(1,-1)
    z = torch.linspace(3.2, 3.8, 5, dtype=torch.float64).view(1,-1)

    testq = qline_diff3D(Q, X0, X1, Sigma,
                                x.unsqueeze(2).unsqueeze(2),
                                y.unsqueeze(2).unsqueeze(1),
                                z.unsqueeze(1).unsqueeze(1))
    with open('exact_qline_gaus.json') as f:
        exact = json.load(f)
    for i in range(testq.shape[1]):
        for j in range(testq.shape[2]):
            for k in range(testq.shape[3]):
                d = abs(testq[0,i,j,k].item()-exact[i][j][k])
                msg = f'difference = {d} at index = ({i},{j},{k}), '\
                    f'(x,y,z)={x[0,i].item(),y[0,j].item(),z[0,k].item()}), '\
                    f'testq = {testq[0,i,j,k].item()}, q from mathematica = {exact[i][j][k]}'
                assert torch.isclose(testq[0,i,j,k], torch.tensor(exact[i][j][k], dtype=torch.float64),
                                     atol=1E-12, rtol=1E-12), msg

def main():
    print('------ test_QModel ------')
    test_QModel()

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
