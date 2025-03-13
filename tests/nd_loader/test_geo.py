import sys
import h5py
import numpy as np
import yaml
from collections import defaultdict
import torch
import time

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
from tred.types import index_dtype

from tred.units import cm, mm

from importlib import reload  # Python 3.4+
import tred.loaders

tred.loaders = reload(tred.loaders)

from tred.loaders import StepLoader, steps_from_ndh5

import tred.io_nd

tred.io_nd = reload(tred.io_nd)

from tred.io_nd import (
    nd_collate_fn, create_tpc_datasets_from_steps,
    LazyLabelBatchSampler, EagerLabelBatchSampler, SortedLabelBatchSampler, CustomNDLoader,
    simple_geo_parser, tpc_label, tpc_drift_direction,
)

def raster(xdata, ydata, ndlow, ndhigh):
    dx = ndhigh[0]-ndlow[0]
    dy = ndhigh[1]-ndlow[1]
    nbins = 200
    xbinwidth = dx / nbins
    ybinwidth = dy / nbins

    xfull = ndlow[0] - 20 *xbinwidth, ndhigh[0] + 20*xbinwidth
    yfull = ndlow[1] - 20 *ybinwidth, ndhigh[1] + 20*ybinwidth

    nbinsfull = 40 + nbins

    H, xedges, yedges = np.histogram2d(xdata, ydata, range=[xfull, yfull], bins=nbinsfull)
    x = (xedges[:-1] + xedges[1:])/2
    y = (yedges[:-1] + yedges[1:])/2
    xpos = []
    ypos = []
    h = []

    for ix, xc in enumerate(x):
        for iy, yc in enumerate(y):
            if H[ix,iy]>0:
                xpos.append(xc)
                ypos.append(yc)
                h.append(H[ix,iy])
    return h, xpos, ypos


def test_drift_direction():
    borders = simple_geo_parser('../nd_geometry/ndlar-module.yaml', '../nd_geometry/multi_tile_layout-3.0.40.yaml')

    anodes, cathodes, drift_directions = tpc_drift_direction(borders)
    plot_anode_cathode(anodes, cathodes, drift_directions)

def plot_anode_cathode(anodes, cathodes, drift_directions, output_file='tpc_drift_plots.pdf'):
    num_entries = anodes.shape[0]
    num_pages = (num_entries + 1) // 2  # Each page contains two entries

    # Create a PDF file to save all figures

    with PdfPages(output_file) as pdf:
        for page in range(num_pages):
            fig, ax = plt.subplots(figsize=(10, 5))  # Single axis for both entries

            for i in range(2):
                index = page * 2 + i
                if index >= num_entries:
                    break

                anode = anodes[index].item()
                cathode = cathodes[index].item()
                drift_direction = drift_directions[index].item()

                # Plot the anode and cathode as vertical lines
                ax.axvline(anode, color='r', linestyle='-', label=f'Anode {index}' if i == 0 else None)
                ax.axvline(cathode, color='b', linestyle='-', label=f'Cathode {index}' if i == 0 else None)

                # Annotate drift direction with an arrow
                arrow_x = (anode + cathode) / 2
                arrow_y = 0.5 - i * 0.2  # Offset arrows slightly to distinguish
                ax.annotate("", xytext=(arrow_x, arrow_y), xy=(arrow_x + 0.9*abs(anode-arrow_x)*drift_direction, arrow_y),
                           arrowprops=dict(arrowstyle="->"))
                # ax.annotate("", xytext=(arrow_x, arrow_y), xy=(anode, arrow_y),
                #            arrowprops=dict(arrowstyle="->"))

            # Labels and title
            ax.set_title(f'Entries {page * 2} & {page * 2 + 1}')
            ax.set_xlim(min(anodes.min(), cathodes.min()) - 20, max(anodes.max(), cathodes.max()) + 20)
            ax.set_ylim(-1, 1)
            ax.set_xlabel('Position x[cm]')
            ax.set_ylabel('Dummy axis')
            ax.legend()
            ax.grid()

            plt.tight_layout()
            pdf.savefig(fig)  # Save the current figure to the PDF
            plt.close(fig)

def test_tpc_dataset():
    borders = simple_geo_parser('../nd_geometry/ndlar-module.yaml', '../nd_geometry/multi_tile_layout-3.0.40.yaml')

    path = '/home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5'
    d0 = StepLoader(h5py.File(path), transform=steps_from_ndh5)

    f32, f64, i32 = d0[:]

    tpcs = create_tpc_datasets_from_steps((f32, f64, i32), i32, borders, sort_index=1)

    ndual = (len(tpcs) + 1)//2

    output_file = 'check_tpc.pdf'
    with PdfPages(output_file) as pdf:
        xlimits = borders[:,0,:].min()-20, borders[:,0,:].max()+20
        ylimits = borders[:,2,:].min()-20, borders[:,2,:].max()+20
        tpclow = borders[:,0,:].min(), borders[:,2,:].min()
        tpchigh = borders[:,0,:].max(), borders[:,2,:].max()
        for it in range(ndual):
            fig, ax = plt.subplots(figsize=(10, 5))  # Single axis for both entries
            for j in range(2):
                tpc = tpcs[2*it+j]
                f0, i0 = tpc[:]
                f0 = f0[0]
                pos = (f0[:,2:5] + f0[:,5:8]).numpy()/2.

                anode = tpc.anode
                cathode = tpc.cathode


                box = borders[2*it+j]
                x = [box[0,0], box[0,0], box[0,1], box[0, 1], box[0,0]]
                y = [box[2,0], box[2,1], box[2,1], box[2,0], box[2,0]]

                ax.plot(x, y, 'o-', label=f'TPC border {2*it+j}')
                ax.axvline(anode, color='r', linestyle='--', label=f'Anode group {it}' if j == 0 else None)
                ax.axvline(cathode, color='b', linestyle='--', label=f'Cathode group {it}' if j == 0 else None)
                arrow_x = (anode + cathode) / 2
                arrow_y = box[2,:].max() + 10 # Offset arrows slightly to distinguish
                ax.annotate("", xytext=(arrow_x, arrow_y), xy=(arrow_x + abs(anode-arrow_x)*tpc.drift.item(), arrow_y),
                           arrowprops=dict(arrowstyle="->"))
                # ax.scatter(x=pos[:,0], y=pos[:,2], s=1, label=f'Event activity in TPC {2*it+j}')
                h, xc, yc = raster(pos[:,0], pos[:,2], tpclow, tpchigh)
                ax.scatter(x=xc, y=yc, s=1, c=h, label=f'Event Activity in TPC {2*it+j}')

            # Labels and title
            ax.set_title(f'Entries {it* 2} & {it* 2 + 1}')
            ax.set_xlim(borders[:,0,:].min()-20, borders[:,0,:].max()+20)
            ax.set_ylim(borders[:,2,:].min()-20, borders[:,2,:].max()+20)
            ax.set_xlim(*xlimits)
            ax.set_ylim(*ylimits)
            ax.set_xlabel('Position x[cm]')
            ax.set_ylabel('Position z[cm]')
            if (it % 5 < 3) and (it //5 <3):
                loc='upper right'
            if it % 5 < 3 and it //5 >=3:
                loc='upper left'
            if it % 5 >= 3 and it //5 <3:
                loc='lower right'
            if it % 5 >= 3 and it //5 >=3:
                loc='lower left'
            loc='best'
            ax.legend()
            ax.grid()

            plt.tight_layout()
            pdf.savefig(fig)  # Save the current figure to the PDF
            plt.close(fig)

def main():
    test_drift_direction()
    test_tpc_dataset()

if __name__ == '__main__':
    main()
