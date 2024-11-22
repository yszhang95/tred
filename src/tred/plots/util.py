#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


