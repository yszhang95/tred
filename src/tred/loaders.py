#!/usr/bin/env python
'''
Data loaders, more properly called "PyTorch Datasets" are iterators that
yield tensors.

Developers take note:

- Do NOT mess with "device" here.  Load to torch.Tensor without specifying a
"device" argument and no calls to Tensor.to().

- It is not required to pass requires_grad=False, but okay to do so.

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


def npz_keys(path):
    return numpy.load(path).keys()

class NpzFile:
    '''
    Dict-like object of tensors from an Numpy npz file.
    '''
    def __init__(self, path, dtype=torch.float32):
        self._path = path
        self._fp = numpy.load(path)
        self._dtype = dtype

    def keys(self):
        return self._fp.keys()

    def __getitem__(self, key):
        return torch.tensor(self._fp[key], dtype=self._dtype, requires_grad=False)

    def get(self, key, default=None, dtype=torch.float32):
        try:
            dat = self._fp[key]
        except KeyError:
            return default
        return torch.tensor(dat, dtype=dtype, requires_grad=False)


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
    def __init__(self, path, dtype=torch.float32):
        self._path = path
        self._fp = h5py.File(path)
        self._dtype = dtype

    def keys(self):
        return hdf_keys(self._fp)

    def __getitem__(self, key):
        return torch.tensor(self._fp[key][:], dtype=self._dtype, requires_grad=False)

    def get(self, key, default=None, dtype=torch.float32):
        try:
            dat = self._fp[key][:]
        except KeyError:
            return default
        return torch.tensor(dat, dtype=dtype, requires_grad=False)


def file_xxx(filepath, dtype=torch.float32):
    '''
    Return a dict-like mapping from string to tensors.

    This detects underlying file format.

    A default dtype may be given.
    '''
    mt = mime_type(filepath)

    if mt == "application/zip":
        return NpzFile(filepath, dtype)

    if mt == "application/x-hdf5":
        return HdfFile(filepath, dtype)


class StepLoader:
    '''
    Produce step tensors assuming data is in Step schema.
    The supported data is in a foramt of h5py.File or numpy structured array.
    The output data is a tuple of a tensor for float and a tensor for integer.
    - The tensor in float type spans (N_batch, 8), by 'dE', 'dEdx', 'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end'.
    - the tensor in integer type spans (N_batch, 2), by 'pdg_id', 'event_id'.

    Random access is only available for batch dimension, not along feature dimension.

    FIXME: Unsigned integers are implicitly converted to signed integers.

    FIXME: Advanced indexing may be supported in the future.

    FIXME: A future version should decouple ND specific format to a transform function
           and make the function derived from PyTorch dataset.
    '''
    DTYPE = {
        'float32' : torch.float32,
        'int32' : torch.int32,
    }

    def __init__(self, data):
        # fixme: the following can be wraped in a transform function specialized to ND
        if isinstance(data, h5py.File) and 'segments' in data.keys():
            data = data['segments']
        self._dkeys = {
            'float32' : ['dE', 'dEdx', 'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end'],
            'int32' : ['pdg_id', 'event_id'] # fixme: event_id is in uint32 at ND
        }

        self._data = {}
        for k, v in self._dkeys.items():
            self._data[k] = torch.stack([torch.tensor(data[n], dtype=StepLoader.DTYPE[k], requires_grad=False) for n in v], dim=1) \
                if isinstance(v, (list, tuple)) else torch.tensor(data[v], dtype=StepLoader.DTYPE[k], requires_grad=False)

    def __len__(self):
        return len(self._data['float32'])

    def __getitem__(self, idx):
        return self._data['float32'][idx], self._data['int32'][idx]

    def get_column(self, key):
        '''
        Get a column by key.
        Args:
            key: the identifier of the column; tt is usually a string.
        Returns:
            A column tensor in a size of (N_batch, ); data type depends on the key.
        Raises:
            ValueError: key must be within available.
        '''
        v = None
        for k, v in self._dkeys.items():
            try:
                idx = v.index(key)
                if len(self._data[k].shape) > 1:
                    v = self._data[k][:,idx]
                    return v
                else:
                    v = self._data[k][:]
                    return v
            except ValueError:
                pass
        if v is None:
            raise ValueError(f'key must be in {self._dkeys}')


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

