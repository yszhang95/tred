from tred.response import ndlarsim
from .util import make_figure
from tred.web import download
import numpy

def plot_ndlarsim(out):
    fname = download('https://www.phy.bnl.gov/~bviren/tmp/tred/response_38_v2b_50ns_ndlar.npy')
    r = ndlarsim(fname)

    fig, ax = make_figure('ND-Lar-sim response\nSelect points along the diagonal\nasymmetric colors imply a problem')

    tick0=6250
    tickl=r.shape[-1]
    ticks=numpy.arange(tick0,tickl)

    rend = r[:,:,tick0:tickl]
    for ind1 in [0, 1, 2, 3, 4, 5, 14, 24, 34, 44]:
        ind2 = 90 - ind1 - 1
        lines = ax.plot(ticks, rend[ind1,ind1,:], label = f'+{ind1} -{ind2}')
        color = lines[0].get_color()
        ax.plot(ticks, -rend[ind2,ind2,:], color=color)
    ax.legend()
    ax.set_xlabel('field response tick near end')
    out.savefig()
    
def plots(out):
    plot_ndlarsim(out)
    
