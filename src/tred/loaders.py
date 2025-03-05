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
from torch import Tensor

from typing import Tuple

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

# FIXME: to use torch.jit.script
def _equal_div(X0X1: Tensor, ExtensiveFloat: Tensor,
               IntensiveFloat: Tensor,
               ExtensiveDouble: Tensor,
               IntensiveDouble: Tensor,
               IntensiveInt: Tensor,
               Ns: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''
    Divide X0X1[i] to Ns[i] sub-segments.
    Divide ExtensiveFloat[i] by Ns[i].
    Repeat IntensiveFloat[i] and IntensiveInt[i] by Ns[i] times.

    FIXME: To find a way couple this function with class StepLoader.
           Now it is outside the class as it requires compilation.
    '''
    batch_size = X0X1.shape[0]
    device = X0X1.device

    # Find the max segment count to unify tensor sizes
    max_size = Ns.max()
    N_LIMIT = 1000
    MIN_STEP = 1/N_LIMIT
    assert max_size + 1 < N_LIMIT

    # Create a common index tensor (0 to max_size)
    idxs = torch.arange(max_size + 0.1, device=device).float()  # Shape: (max_size + 1,)

    # Normalize indices based on Ns (broadcasting)
    frac_steps = idxs[None, :] / Ns[:, None]  # Shape: (batch_size, max_size+1)

    # Compute interpolated values using broadcasting
    xs = (1 - frac_steps) * X0X1[:, 0, None] + frac_steps * X0X1[:, 3, None]
    ys = (1 - frac_steps) * X0X1[:, 1, None] + frac_steps * X0X1[:, 4, None]
    zs = (1 - frac_steps) * X0X1[:, 2, None] + frac_steps * X0X1[:, 5, None]

    # Mask to ignore extra indices for each row
    mask = frac_steps < (1-MIN_STEP) # discard the last point as the lenght of output is less than the lenght of linspace by 1.
    selected = mask[:, :-1]

    # Stack and reshape for output format
    X0X1 = torch.stack([xs[:, :-1], ys[:, :-1], zs[:, :-1], xs[:, 1:], ys[:, 1:], zs[:, 1:]], dim=-1)
    X0X1 = X0X1[selected].view(-1, 6)  # Remove invalid entries

    if ExtensiveFloat.dim() == 1:
        ExtensiveFloat = ExtensiveFloat.unsqueeze(1)
    if IntensiveFloat.dim() == 1:
        IntensiveFloat = IntensiveFloat.unsqueeze(1)
    if ExtensiveDouble.dim() == 1:
        ExtensiveDouble = ExtensiveDouble.unsqueeze(1)
    if IntensiveDouble.dim() == 1:
        IntensiveDouble = IntensiveDouble.unsqueeze(1)
    if IntensiveInt.dim() == 1:
        IntensiveInt = IntensiveInt.unsqueeze(1)

    ExtensiveFloat = ExtensiveFloat / Ns[:,None]
    ExtensiveFloat = ExtensiveFloat.repeat(1,max_size).view(batch_size,
                                                            max_size, -1)[selected]
    ExtensiveDouble = ExtensiveDouble / Ns[:,None]
    ExtensiveDouble = ExtensiveDouble.repeat(1,max_size).view(batch_size,
                                                            max_size, -1)[selected]

    IntensiveFloat = IntensiveFloat.repeat(1,max_size).view(batch_size, max_size, -1)[selected]
    IntensiveDouble = IntensiveDouble.repeat(1,max_size).view(batch_size, max_size, -1)[selected]
    IntensiveInt = IntensiveInt.repeat(1,max_size).view(batch_size, max_size, -1)[selected]

    return X0X1, ExtensiveFloat, IntensiveFloat, ExtensiveDouble, IntensiveDouble, IntensiveInt

_equal_div_script = torch.jit.script(_equal_div)

def steps_from_ndh5(data, type_map=None):
    '''
    Arguments:
        data: The supported data is in a foramt of h5py.File or numpy structured array.
        type_map: type string to data type in pytorch
    Output:
        The output data is a tuple of a tensor for float and a tensor for integer.
        - The tensor in float type spans (N_batch, 8), by 'dE', 'dEdx', 'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end'.
        - The tensor in double type spans (N_batch, ) by 't0'.
        - the tensor in integer type spans (N_batch, 2), by 'pdg_id', 'event_id'.

    FIXME: Unsigned integers are implicitly converted to signed integers.
    '''
    if type_map is None:
        type_map = {
            'float32' : torch.float32,
            'float64' : torch.float64,
            'int32' : torch.int32,
        }
    if isinstance(data, h5py.File) and 'segments' in data.keys():
        data = data['segments']
    _dkeys = {
            'float32' : ['dE', 'dEdx', 'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end'],
            'float64' : ['t0_start'],
            'int32' : ['event_id', 'vertex_id', 'pdg_id'] # fixme: event_id is in uint32 at ND
    }

    _data = {}
    for k, v in _dkeys.items():
        # FIXME: check data[n].flags['C_CONTIGUOUS']) and force to convert to contiguous array
        # data[n] = np.ascontiguousarray(data[n])
        _data[k] = torch.stack([torch.tensor(data[n].copy(), dtype=type_map[k], requires_grad=False) for n in v], dim=1) \
            if isinstance(v, (list, tuple)) else torch.tensor(data[v], dtype=type_map[k], requires_grad=False)

    return _dkeys, _data

class StepLoader:
    '''
    Produce step tensors assuming data is in Step schema.
    The supported data is in a foramt of h5py.File or numpy structured array.
    The output data is a tuple of a tensor for float, double and a tensor for integer.
    - The tensor in float type spans (N_batch, 8), by 'dE', 'dEdx', 'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end'.
    - The tensor in double type spans (N_batch, ) by 't0'.
    - the tensor in integer type spans (N_batch, 2), by 'pdg_id', 'event_id'.

    Random access is only available for batch dimension, not along feature dimension.

    FIXME: Advanced indexing may be supported in the future.
    FIXME: the dataset does not provide `labels` array.
    '''
    DTYPE = {
        'float32' : torch.float32,
        'float64' : torch.float64,
        'int32' : torch.int32,
    }

    def __init__(self, data, transform, target_transform=None, **kwargs):
        # fixme: the following can be wraped in a transform function specialized to ND
        self._dkeys, self._data = transform(data, StepLoader.DTYPE)

        # preprocessing
        step_limit = kwargs.get('step_limit', 1)
        mem_limit = kwargs.get('mem_limit', 1024) # MB
        device = kwargs.get('device', 'cpu')
        fn = kwargs.get('preprocesing', _equal_div_script)

        X0X1 = self._data['float32'][:,2:8]
        dE = self._data['float32'][:,0]
        dEdx = self._data['float32'][:,1]
        IntensiveInt = self._data['int32']
        ExtensiveDouble = torch.empty((len(dE),0), requires_grad=False, dtype=torch.float64)
        IntensiveDouble = self._data['float64'][:]
        X0X1, dE, dEdx, _, IntensiveDouble, IntensiveInt = (
            StepLoader._batch_equal_div(X0X1, dE, dEdx, ExtensiveDouble, IntensiveDouble,
                                        IntensiveInt,
                                        step_limit, mem_limit, device, fn
                                        )
        )

        self._data['float32'] = torch.cat([dE.view(-1,1), dEdx.view(-1,1), X0X1], dim=1)
        self._data['float64'] = IntensiveDouble
        self._data['int32'] = IntensiveInt
        self.length = len(self._data['int32'])

    @staticmethod
    def _batch_equal_div(X0X1, ExtensiveFloat, IntensiveFloat, ExtensiveDouble,
                        IntensiveDouble, IntensiveInt,
                         step_limit, mem_limit, device='cpu', fn=_equal_div_script):
        old_device = X0X1.device
        X0X1 = X0X1.to(device)
        ExtensiveFloat = ExtensiveFloat.to(device)
        IntensiveFloat = IntensiveFloat.to(device)
        ExtensiveDouble = ExtensiveDouble.to(device)
        IntensiveDouble = IntensiveDouble.to(device)
        IntensiveInt = IntensiveInt.to(device)

        n_limit = mem_limit * 1024 * 1024 # Bytes
        LdX = torch.linalg.norm(X0X1[:,3:]-X0X1[:,:3], dim=1)
        Ns = (LdX//step_limit + 1).to(torch.int32)
        max_size = Ns.max().item()
        nextf = 1 if len(ExtensiveFloat.shape) == 1 else ExtensiveFloat.size(1)
        nintf = 1 if len(IntensiveFloat.shape) == 1 else IntensiveFloat.size(1)
        nextd = 1 if len(ExtensiveDouble.shape) == 1 else ExtensiveDouble.size(1)
        nintd = 1 if len(IntensiveDouble.shape) == 1 else IntensiveDouble.size(1)
        ninti = 1 if len(IntensiveInt.shape) == 1 else IntensiveInt.size(1)
        nbytes = max_size * X0X1.size(0) * (X0X1.size(1) + nextf + nintf + 2*nextd + 2*nintd + ninti) * 4 # Bytes
        n_chunk = int(nbytes // n_limit) + 1

        Nss = Ns.chunk(n_chunk)
        X0X1s = X0X1.chunk(n_chunk)
        ExtensiveFloats = ExtensiveFloat.chunk(n_chunk)
        IntensiveFloats = IntensiveFloat.chunk(n_chunk)
        ExtensiveDoubles = ExtensiveDouble.chunk(n_chunk)
        IntensiveDoubles = IntensiveDouble.chunk(n_chunk)
        IntensiveInts = IntensiveInt.chunk(n_chunk)

        tensors = []
        for (_x0x1, _extf, _intf, _extd,
             _intd, _inti, _ns) in zip(X0X1s, ExtensiveFloats, IntensiveFloats,
                                       ExtensiveDoubles, IntensiveDoubles,
                                       IntensiveInts, Nss):
            args = (_x0x1, _extf, _intf, _extd, _intd, _inti, _ns)
            tensors.append(
                fn(*args)
            )
            tensors[-1] = tuple(t.to(old_device) for t in tensors[-1])
        return tuple(torch.cat([tensors[i][j] for i in range(len(tensors))], dim=0)
                     for j in range(len(tensors[0])))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self._data['float32'][idx], self._data['float64'][idx], self._data['int32'][idx]

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

