#!/usr/bin/env python

import os

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Circle
except ImportError:
    print("\nNo matplotlib available, consider running 'uv sync --extra matplotlib'\n")
    raise
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, SymLogNorm

class NameSequence(object):

    def __init__(self, name, first=0, **kwds):
        '''
        Every time called, emit a new name with an index.

        Name may be a filename with .-separated extension.

        The name may include zero or one %-format mark which will be
        fed an integer counting the number of output calls.

        If no format marker then the index marker is added between
        file base and extension (if any).

        The first may give the starting index sequence or if None no
        sequence will be used and the name will be kept as-is.

        Any keywords will be applied to savefig().

        This is a callable and it mimics PdfPages.
        '''
        self.base, self.ext = os.path.splitext(name)
        self.index = first
        self.opts = kwds

    def __call__(self):
        if self.index is None:
            return self.base + self.ext

        try:
            fn = self.base % (self.index,)
        except TypeError:
            fn = '%s%04d' % (self.base, self.index)
        self.index += 1

        ret = fn + self.ext
        return ret

    def savefig(self, *args, **kwds):
        '''
        Act like PdfPages
        '''
        opts = dict(self.opts, **kwds)

        fn = self()
        dirn = os.path.dirname(fn)
        if dirn and not os.path.exists(dirn):
            os.makedirs(dirn)
        plt.savefig(fn, **opts)


    def __enter__(self):
        return self
    def __exit__(self, typ, value, traceback):
        return
        


def pages(name, fmt=None):
    '''
    https://matplotlib.org/stable/gallery/misc/multipage_pdf.html
    '''
    if name.endswith(".pdf") or fmt == "pdf":
        return PdfPages(name)
    return NameSequence(name)
    


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

