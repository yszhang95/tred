#!/usr/bin/env python
'''
Data loaders are iterators that yield tensors.

'''
import torch
from .util import mime_type

# I/O techs are optional.  If not installed, delay error until use
try:
    import numpy
except ImportError:
    numpy = None
try:
    import h5py
except ImportError:
    h5py = None

import logging
log = logging.getLogger("tred")


def npz_keys(path):
    return numpy.load(path).keys()

class NpzFile:
    '''
    Dict-like object of tensors from an Numpy npz file.
    '''
    def __init__(self, path, dtype=torch.float32, device='cpu'):
        self._path = path
        self._fp = numpy.load(path)
        if device == 'gpu':
            device = 'cuda'
        self._device = device
        self._dtype = dtype

    def keys(self):
        return self._fp.keys()

    def __getitem__(self, key):
        return torch.tensor(self._fp[key], dtype=self._dtype, device=self._device, requires_grad=False)

    def get(self, key, default=None, dtype=torch.float32, device='cpu'):
        try:
            dat = self._fp[key]
        except KeyError:
            return default
        return torch.tensor(dat, dtype=dtype, device=device, requires_grad=False)


def hdf_keys(obj):
    "Recursively find all dataset keys"

    if isinstance(obj, str):
        obj = h5py.File(fobj)

    if not isinstance(obj, h5py.Group):
        return (obj.name,)

    keys = ()                   # don't emit key to group
    for key, value in obj.items():
        keys = keys + hdf_keys(value)

    return keys


class HdfFile:
    '''
    Dict-like object of tensors from an HDF5 file.
    '''
    def __init__(self, path, dtype=torch.float32, device='cpu'):
        self._path = path
        self._fp = h5py.File(path)
        if device == 'gpu':
            device = 'cuda'
        self._device = device
        self._dtype = dtype

    def keys(self):
        return hdf_keys(self._fp)

    def __getitem__(self, key):
        return torch.tensor(self._fp[key][:], dtype=self._dtype, device=self._device, requires_grad=False)

    def get(self, key, default=None, dtype=torch.float32, device='cpu'):
        try:
            dat = self._fp[key][:]
        except KeyError:
            return default
        return torch.tensor(dat, dtype=dtype, device=device, requires_grad=False)


def file_xxx(filepath, dtype=torch.float32, device='cpu'):
    '''
    Return a dict-like mapping from string to tensors.

    This detects underlying file format.
    '''
    mt = mime_type(path)

    if mt == "application/zip":
        return NpzFile(path, dtype, device)

    if mt == "application/x-hdf5":
        return HdfFile(path, dtype, device)


class DepoLoader:
    '''
    Produce depo tensors assuming data is in depo data/info schema.

    The input depo_data array is shaped (7,n) and depo_info (4,n) but in some
    old cases the transpose.  The 7 dimension spans: (t,q,x,y,z,sigmaL,sigmaT).
    The 4 dimension spans: (id, pdg, gen, child).

    This will remove all but the youngest depos (gen>0)
    '''

    def __init__(self, data):
        self._dkeys = [k for k in data.keys() if 'depo_data_' in k]
        self._data = data

    def __len__(self):
        return len(self._dkeys)

    def __getitem__(self, idx):
        dkey = self._dkeys[idx]
        data = self._data.get(dkey, dtype=torch.float32)
        info = self._data.get(dkey.replace('depo_data_', 'depo_info_'), dtype=torch.int32)
        if not (data.shape[0] == 7 and info.shape[0] == 4):
            data = data.T
            info = info.T
        youngest = info[2] == 0
        return data[:,youngest]

    def __iter__(self):
        for idx in range(len(self._dkeys)):
            yield self[idx]


def make_depos(data):
    '''
    Return depo loader if this looks like depo data
    '''
    nd=0
    ni=0
    for key in data.keys():
        if key.startswith("depo_data_"): nd += 1
        if key.startswith("depo_info_"): ni += 1
    if nd == ni and nd > 0:
        return DepoLoader(data)
    
    

def make_xxx(data):
    '''
    Convert keyed data to a iterative loader.
    '''
    g = globals()

    # fixme: steps, blocks, ....
    for schema in ["depos", ]:
        meth = g[f'make_{schema}']
        loader = make_depos(data)
        if loader:
            return loader


def load(path):
    '''
    Return list-like object to iterate data in file.

    This function tries to auto-detect the format.
    '''
    # fixme: handle multiple files.

    data = file_xxx(path)
    return make_xxx(data)
