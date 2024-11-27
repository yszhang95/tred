#!/usr/bin/env python

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Circle
except ImportError:
    print("\nNo matplotlib available, consider running 'uv sync --extra matplotlib'\n")
    raise
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

def pages(name, fmt=None):
    '''
    For now, just PDF.
    https://matplotlib.org/stable/gallery/misc/multipage_pdf.html
    '''
    if name.endswith(".pdf") or fmt == "pdf":
        return PdfPages(name)
    raise ValueError(f'not yet support for {name} {fmt or ""}')


def make_figure(title, nrows=1, ncols=1, **kwds):
    '''
    Standardize making a figure.
    '''
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **kwds)
    fig.suptitle(title)
    return fig, axes


def ellipse(center, sigma, **kwds):
    return Ellipse((center[1],center[0]), height=sigma[0], width=sigma[1], **kwds)

def circle(center, radius, **kwds):
    return Circle((center[1],center[0]), radius, **kwds)

