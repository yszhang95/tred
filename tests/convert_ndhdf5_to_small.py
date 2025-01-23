#!/usr/bin/env python3
import h5py

import numpy as np

import os
import sys

def main(fpath, outdir=None):
    f = h5py.File(fpath, 'r')
    if outdir is None:
        outdir = os.path.dirname(fpath)
    f1 = h5py.File(os.path.join(outdir, 'segments_first_event.hdf5'), 'w')
    f2 = h5py.File(os.path.join(outdir, 'segments_a_muon_first_event.hdf5'), 'w')
    segments = f['segments']
    event_id = segments['event_id']
    id0 = np.unique(event_id)[0]
    event0 = segments[event_id == id0]
    pdgid_mu = 13
    musegments = event0[event0['pdg_id'] == pdgid_mu]
    muvertices = np.unique(musegments['vertex_id'])
    single_mu = musegments[musegments['vertex_id'] == muvertices[0]]
    f1.create_dataset('segments', data=event0)
    f2.create_dataset('segments', data=single_mu)


if __name__ == '__main__':
    try:
        f = sys.argv[1]
    except IndexError:
        exit(-1)
    main(f)
