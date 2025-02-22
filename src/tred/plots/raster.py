import math
import torch
import tred.raster as tr

from tred.plots.util import make_figure, Ellipse, Circle, LogNorm
from tred.util import make_points
from tred.drift import drift
from tred.raster import depos

def make_3d():
    vdim = 3                    # 3 spatial (vector) dimensions

    locs = torch.tensor([
        [-10.0, -5.0, 0.0, 5.0, 10.0, 20], # x
        [-20.0, -10.0, 0.0, 10.0, 20.0, 40], # y
        [-20.0, -10.0, 0.0, 10.0, 20.0, 40], # z
    ]).T

    velocity = -1               # from high location value to low
    target = 0
    diffusion = torch.tensor([1,2,2])
    lifetime = 10
    nsigma = torch.tensor([3.0]*vdim)
    
    centers, times, sigmas, charges = drift(locs, velocity, diffusion, lifetime)

    centers[:,0] = times        # from (x,y),t to (t,y),x=x_resp
    sigmas[:0] /= abs(velocity) # sigma_x to sigma_t

    tgrid = 1
    ygrid = 2
    zgrid = 2

    grid = torch.tensor([tgrid, ygrid, zgrid])

    rasters, offsets = depos.binned(grid, centers, sigmas, charges, nsigma=nsigma)
    return grid, rasters, offsets


def sum_3d_serial(rasters, offsets):
    '''
    Do a serial sum of offset rasters and return array spanning grid.
    '''
    rshape = torch.tensor(rasters.shape[1:])
    ends = offsets + rshape + 1 # add one to show we aren't truncating raster

    mincorner,_ = torch.min(offsets, dim=0)
    maxcorner,_ = torch.max(ends, dim=0)

    full_shape = maxcorner - mincorner
    full_raster = torch.zeros(tuple(full_shape.tolist()))

    for r, o in zip(rasters, offsets):
        b = o - mincorner
        e = b + rshape
        full_raster[b[0]:e[0], b[1]:e[1], b[2]:e[2]] += r
    return full_raster


def sum_3d(rasters, offsets):
    '''
    Sum rasters given offsets
    '''
    # 1. transform (ndepo, 3) offsets to 

    # tot = torch.zeros_like(rasters[0])
    # tot.index_put_(tuple(offsets.T), rasters, accumulate=True)
    # return tot


def plot_depo_k3d():
    '''
    Generate some 3D depos, drift, raster and plot with k3d.

    Note, this MUST be run from jupyter notebook.

    $ uv run --with jupyter jupyter notebook

    Then in a notebook cell:
    
    import tred.plots.raster
    tred.plots.raster.plot_depo_k3d()
    '''
    import k3d
    grid, rasters, offsets = make_3d()
    full_raster = sum_3d_serial(rasters, offsets)

    vol = k3d.volume(full_raster)
    plot = k3d.plot()
    plot += vol
    plot.display()


def plot_depo_3d(out):
    '''
    Generate some 3D depos, drift, raster and plot.
    '''
    grid, rasters, offsets = make_3d()

    full_raster = sum_3d_serial(rasters, offsets)

    fig, axes = make_figure('3D depo rasters', nrows=3)

    for axis in range(3):
        ax = axes[axis]
        two = torch.sum(full_raster, axis=axis)
        im = ax.imshow(two, aspect='auto',
                       norm=LogNorm(vmin=0.01, vmax=torch.max(full_raster)))
        ax.set_title(f'sum over axis {axis}')

        fig.colorbar(im)
    fig.tight_layout()

    out.savefig()

def plot_depo_2d(out):
    '''
    Generate some 2D depos, drift, raster and plot.
    '''
    vdim = 2                    # 2 spatial (vector) dimensions

    locs = torch.tensor([
        [-10.0, -5.0, 0.0, 5.0, 10.0, 20], # x
        [-20.0, -10.0, 0.0, 10.0, 20.0, 40], # y
    ]).T

    velocity = -1               # from high location value to low
    target = 0
    diffusion = torch.tensor([1,2])
    lifetime = 10
    nsigma = torch.tensor([3.0]*vdim)
    
    centers, times, sigmas, charges = drift(locs, velocity, diffusion, lifetime)

    centers[:,0] = times        # from (x,y),t to (t,y),x=x_resp
    sigmas[:0] /= abs(velocity) # sigma_x to sigma_t

    tgrid = 1
    ygrid = 2

    grid = torch.tensor([tgrid, ygrid])
    
    rasters, offsets = depos.binned(grid, centers, sigmas, charges, nsigma=nsigma)
    rshape = torch.tensor(rasters.shape[1:])
    ends = offsets + rshape + 1 # add one to show we aren't truncating raster

    mincorner,_ = torch.min(offsets, dim=0)
    maxcorner,_ = torch.max(ends, dim=0)

    full_shape = maxcorner - mincorner
    full_raster = torch.zeros(tuple(full_shape.tolist()))

    for r, o in zip(rasters, offsets):
        b = o - mincorner
        e = b + rshape
        full_raster[b[0]:e[0], b[1]:e[1]] += r

    minpt = (mincorner - 0.5) * grid
    maxpt = (maxcorner - 0.5) * grid

    fig, ax = make_figure('2D depo rasters')
    im = ax.imshow(full_raster.T,
                   origin = 'lower',
                   extent = (minpt[0],maxpt[0],minpt[1],maxpt[1]),
                   norm=LogNorm(vmin=0.01, vmax=torch.max(full_raster)))

    ## to check image coordinate system
    # ax.add_patch(Circle((0,0), 1, fill=True, color="red"))
    # ax.add_patch(Circle((10,0), 1, fill=True, color="green"))
    # ax.add_patch(Circle((0,20), 1, fill=True, color="blue"))

    for center,sigma,charge in zip(centers,sigmas,charges):
        # center = center / grid - mincorner
        if sigma[0] == 0 or sigma[1] == 0:
            cir = Circle(center, 1.0, fill=False)
            ax.add_patch(cir)
            continue
        ell = Ellipse(center, width=sigma[0], height=sigma[1], fill=False)
        ax.add_patch(ell)
    ax.plot((0,0), (minpt[1], maxpt[1]))
    ax.set_xlabel('drift time')
    ax.set_ylabel('pitch location')    
    fig.colorbar(im)

    out.savefig()

def plot_depo_1d(out):
    '''
    Generate some 1D depos, drift, raster and plot.
    '''
    vdim = 1

    locs = torch.tensor([-1.0, 0.0, 1.0, 4.0, 8.0])

    velocity = -1               # from high location value to low
    target = 0
    diffusion = 1
    lifetime = 10
    nsigma = 3.0
    
    centers, times, widths, charges = drift(locs, velocity, diffusion, lifetime)

    grid = torch.tensor([1.0])

    # in 1D the single spatial drift coordinate becomes the drift time coordinate
    rasters, offsets = depos.binned(grid, times, widths, charges, nsigma=nsigma)

    # find a global range that holds all.
    minind = torch.floor(torch.min(times - nsigma*widths)/grid).to(dtype=torch.int32)[0]
    maxind = torch.ceil(torch.max(times + nsigma*widths)/grid).to(dtype=torch.int32)[0]

    gmin = minind*grid[0]
    gmax = maxind*grid[0]

    # span the plot range
    ls = torch.linspace(gmin, gmax, maxind-minind+1)
    ls0 = ls[0]

    fig, ax = make_figure('1D depo rasters')
    for c,w,q,r,o in zip(times, widths, charges, rasters, offsets):

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

    ax.set_xlabel("drift time")
    ax.set_ylabel("charge")

    out.savefig()



def plots(out):
    glb = globals()
    for d in [1,2,3]:
        meth = glb[f'plot_depo_{d}d']
        meth(out)

