import math
import torch
import tred.raster as tr

from tred.plots.util import make_figure
from tred.util import make_points
from tred.drift import drift
from tred.raster import depo_binned

def plot_depo_1d_1(out):
    xmin, xmax = -1, 10
    npoints = 3
    vdim = 1
    locs = torch.tensor([-1.0, 0.0, 1.0, 4.0, 8.0])

    velocity = -1               # from high location value to low
    target = 0
    D = 1
    lifetime = 10
    nsigma = 3.0
    
    centers, widths, charges = drift(locs, target, velocity, D, lifetime)

    grid = torch.tensor([1.0])

    rasters, offsets = depo_binned(grid, centers, widths, charges, nsigma=nsigma)

    # find a global range that holds all.
    minind = torch.floor(torch.min(centers - nsigma*widths)/grid).to(dtype=torch.int32)[0]
    maxind = torch.ceil(torch.max(centers + nsigma*widths)/grid).to(dtype=torch.int32)[0]

    gmin = minind*grid[0]
    gmax = maxind*grid[0]

    # span the plot range
    ls = torch.linspace(gmin, gmax, maxind-minind+1)
    ls0 = ls[0]

    fig, ax = make_figure('1D rasters')
    for c,w,q,r,o in zip(centers, widths, charges, rasters, offsets):

        g = q*torch.exp(-(ls-c)**2/(2*w**2))/(math.sqrt(2*math.pi)*w)

        oind = o.item()
        nr = len(r)
        lsr = torch.arange(oind*grid[0], (oind+nr)*grid[0], grid[0])

        lines = ax.plot(lsr, r, drawstyle='steps-mid')
        color = lines[0].get_color()

        kwds = dict(color=color, linewidth=0.1)
        if w == 0:
            w = 1
        ax.plot(ls, g, **kwds)
        ax.plot([c,c],[0,q/w], **kwds)
        ax.plot([c,c+w], [q/w,q/w], **kwds)
        ax.plot([c,c-w], [q/w,q/w], **kwds)


    out.savefig()



def plots(out):
    glb = globals()
    for d in [1,]:
        meth = glb[f'plot_depo_{d}d_1']
        meth(out)

