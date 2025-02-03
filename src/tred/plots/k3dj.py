#!/usr/bin/env python
'''
Plotting functions for k3d-jupyter.

These must be run inside jupyter.

 $ ssh -L 8888:localhost:8888 wcgpu0
 $ uv run --with k3d --with jupyter jupyter notebook

         http://localhost:8888/tree?token=<token>

Now open the URL in a local browser, import this module and call a function.
'''


import k3d
import numpy

def voxels(arr):
    vmax = arr.max()
    v = (255*(arr/vmax)).astype(dtype=numpy.uint8)
    vp = k3d.voxel(v, outlines=True, opacity=0.1)
    plot = k3d.plot()
    plot += vp
    plot.display()
    
    
