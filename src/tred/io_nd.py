'''
For usage of DataLoader in PyTorch, check https://github.com/pytorch/pytorch/issues/71872
'''

import yaml
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, BatchSampler

from tred.types import index_dtype
from tred.types import index_dtype
from tred.units import cm, mm

from importlib import reload  # Python 3.4+
import tred.loaders

def simple_geo_parser(det_yaml, tile_yaml):
    '''
    FIXME: slight difference between GDML and YAMLs

    FIXME: output is in units of cm.
    FIXME: consistent usage of units
    '''
    with open(det_yaml, 'r') as f:
        detprop = yaml.safe_load(f)
    with open(tile_yaml, 'r') as f:
        tile_layout = yaml.safe_load(f)
            
    PIXEL_PITCH = tile_layout['pixel_pitch'] * mm / cm
    chip_channel_to_position = tile_layout['chip_channel_to_position']

    TILE_BORDERS = np.zeros((2,2))

    zs = np.array(list(chip_channel_to_position.values()))[:,0] * PIXEL_PITCH
    ys = np.array(list(chip_channel_to_position.values()))[:,1] * PIXEL_PITCH
    TILE_BORDERS[0] = [-(max(zs) + PIXEL_PITCH)/2, (max(zs) + PIXEL_PITCH)/2]
    TILE_BORDERS[1] = [-(max(ys) + PIXEL_PITCH)/2, (max(ys) + PIXEL_PITCH)/2]

    tile_indeces = tile_layout['tile_indeces']
    TILE_POSITIONS = tile_layout['tile_positions']
    tpc_ids = np.unique(np.array(list(tile_indeces.values()))[:,0], axis=0)

    anodes = defaultdict(list)
    for tpc_id in tpc_ids:
        for tile in tile_indeces:
            if tile_indeces[tile][0] == tpc_id:
                anodes[tpc_id].append(TILE_POSITIONS[tile])

    DRIFT_LENGTH = detprop['drift_length']

    TPC_OFFSETS = np.array(detprop['tpc_offsets'])

    TPC_BORDERS = np.empty((TPC_OFFSETS.shape[0] * tpc_ids.shape[0], 3, 2))

    for it, tpc_offset in enumerate(TPC_OFFSETS):
        for ia, anode in enumerate(anodes):
            tiles = np.vstack(anodes[anode]) * mm /cm
            drift_direction = 1 if anode == 1 else -1
            z_border = min(tiles[:,2]) + TILE_BORDERS[0][0] + tpc_offset[2], \
                       max(tiles[:,2]) + TILE_BORDERS[0][1] + tpc_offset[2]
            y_border = min(tiles[:,1]) + TILE_BORDERS[1][0] + tpc_offset[1], \
                       max(tiles[:,1]) + TILE_BORDERS[1][1] + tpc_offset[1]
            # first dimension of tiles is fixed, 0->-46.788cm, 1->46.788cm
            x_border = min(tiles[:,0]) + tpc_offset[0], \
                       max(tiles[:,0]) + DRIFT_LENGTH * drift_direction + tpc_offset[0]
            TPC_BORDERS[it*2+ia] = (x_border, y_border, z_border)
    return torch.tensor(TPC_BORDERS, requires_grad=False)

def tpc_label(borders, X0, X1=None, **kwargs):
    '''
    Args:
        borders: tensor, in a shape of (N_batch, 3, 2),
                 the second dimension for (x, y, z) in order,
                the third dimension is for min or max, without sorting.
        X0: Tensor[float32]
        X1: Tensor[float32] (optional)
        Keyword args:
            - 'nchunk': int, number of TPCs in parallel when selecting segments
    return:
        labels of tpc, tensor, (N_batch,)

    Pre-requisites: there is no overlaps between TPC volumes and steps.
    '''
    labels = torch.full((len(X0),), len(borders), dtype=index_dtype)

    nchunk =  kwargs.get('nchunk', 10)
    X0cs = torch.chunk(X0, nchunk)
    if X1 != None:
        X1cs = torch.chunk(X1, nchunk)
    else:
        X1cs = [None] * len(X0cs)

    labels = []
    
    boxes = borders
    vmin = torch.min(boxes, dim=2)[0] # (ntpc, 3)
    vmax = torch.max(boxes, dim=2)[0] # (ntpc, 3)
    
    for X0c, X1c in zip(X0cs, X1cs):
        labelc = torch.full((len(X0c),), -1, dtype=index_dtype)
        if X1c is None:
            X = X0c
        else:
            X = (X0c + X1c)/2

        c0 = torch.all(X.unsqueeze(1) > vmin[None,...], dim=2) # (npts, 1, 3) > (1, ntpc, 3)
        c1 = torch.all(X.unsqueeze(1) < vmax[None,...], dim=2) # (npts, 1, 3) < (1, ntpc, 3)
        cond = c0 & c1
        #FIXME: assume there is no overlaps
        indices = torch.nonzero(cond, as_tuple=True)
        labelc[indices[0]] = indices[1].to(index_dtype)
        labels.append(labelc)
    labels = torch.concatenate(labels, dim=0)
    return labels

def tpc_drift_direction(borders):
    '''
    assume drift direction to be [-1, 1, -1, 1, ...] alternatively
    assume drift direction is axis0
    --> assume anode position is on borders[:,0,0]
    '''
    # check for conssitency with tpc_label
    boxes = borders
    vmin = torch.min(boxes, dim=2)[0] # (ntpc, 3)
    vmax = torch.max(boxes, dim=2)[0] # (ntpc, 3)

    anodes = []
    cathodes = []
    drift_directions = []

    for ibox, box in enumerate(borders):
        b0 = torch.min(box, dim=1)[0] # (3, )
        b1 = torch.max(box, dim=1)[0] # (3, )
        assert b0.allclose(vmin[ibox], rtol=1E-6, atol=1E-6), 'TPC order in tpc_drift_direction does not match that in tpc_label'
        assert b1.allclose(vmax[ibox], rtol=1E-6, atol=1E-6), 'TPC order in tpc_drift_direction does not match that in tpc_label'
        if ibox %2  == 0:
            assert box[0,0] < box[0,1], 'Expect borders[i,0,0] < borders[i,0,1] for even i so that borders[i,0,0] is the anode position and the drift position is -1.'
            drift_directions.append(-1)
        else:
            assert box[0,0] > box[0,1], 'Expect borders[i,0,0] > borders[i,0,1] for odd i so that borders[i,0,0] is the anode position and the drift position is 1.'
            drift_directions.append(1)
        anodes.append(box[0,0])
        cathodes.append(box[0,1])
    return torch.stack(anodes), torch.stack(cathodes), torch.tensor(drift_directions, requires_grad=False)

def check_features(features):
    '''
    The feature array is assumed to be a list of ntuple, with (Tensor[float32], Tensor[float64], Tensor[int32]).
    The first dimension of each tensor is batch dimension.
    '''
    msg = r'features for TPCDataset must a tuple/list consiting of Tensor[float32], Tensor[float64], Tensor[int32].'
    if not isinstance(features, (tuple,list)):
        raise ValueError(f'{msg} Type of features is {type(features)}.')
    if len(features) != 3:
        raise ValueError(f'{msg} Length of features is {len(features)}.')
    for i, (f, dtype) in enumerate(zip(features, [torch.float32, torch.float64, torch.int32])):
        if f.dtype != dtype:
            raise ValueError(f'{msg} Data type {f.dtype} of features[{i}] does not match {dtype}')
    return

class TPCDataset(Dataset):
    '''
    The feature array is assumed to be a list of ntuple, with (Tensor[float32], Tensor[float64], Tensor[int32]).
    The first dimension of each tensor is batch dimension.
    The label array is an integer array for entry labels and TPC labels. TPC labels are optional.
    '''
    def __init__(self, features, labels, tpc_id, anode=0, cathode=0, drift=0, tpc_label_index=None, sort_index=None):
        '''
        `features` and `labels` are selected according to tpc_id if `labels` consists of the TPC labels as the second label.
        Otherwise, all data in `features` and `labels` are saved.
        Args:
            features: tuple/list of tensor.
            labels: tensor, (N_batch, m) with (label, other_labels, ..., tpclabel) per entry or (N_batch, ) with (label,) per entry
                     Note: the tpc_label_index should not be used when labels.dim() == 1.
            tpc_id: scalar
            tpc_label_index, tpclabels are labels[:,tpclabel_index]
        '''

        check_features(features)

        if labels.dim() != 1 and tpc_label_index != None:
            mask = torch.nonzero(labels[:,tpc_label_index] == tpc_id, as_tuple=True)
            features = [f[mask] for f in features]
            labels = labels[mask]
        else:
            tpc_label_index = None

        if sort_index != None:
            sort_label = None
            if sort_index is True and label.dim() == 1:
                sort_label = label
            elif isinstance(sort_index, int):
                sort_label = label[:, sort_index]
            else:
                raise ValueError
            if sort_label != None:
                indices = torch.argsort(labels, stable=True)
                labels = labels[indices]
                features = [f[indices] for f in  features]

        self.features = features
        if tpc_label_index is None:
            self.labels = labels
        else:
            col_to_remove = tpc_label_index % labels.shape[1]
            self.labels = labels[:, torch.arange(labels.shape[1]) != col_to_remove]

        self.length = len(self.labels)
        self.tpc_id = tpc_id

        self.anode = anode
        self.cathode = cathode
        self.drift = drift

    def __getitem__(self, idx):
        return (self.features[0][idx], self.features[1][idx], self.features[2][idx]), self.labels[idx]

    def __len__(self):
        return self.length

def create_tpc_datasets_from_steps(features, labels, borders, **kwargs):
    '''
    features is a list of (Tensor[float32], Tensor[float64], Tensor[int32]).
    steps is a Tensor[float32], with (N_batch, 8). It corresponds to `StepLoader.__getitem__`.

    labels are assumed to be from 0 to ntpc

    FIXME: to match label with index
    '''
    check_features(features)
    steps = features[0]
    if steps.size(1) != 8:
        raise ValueError('steps must be a size of (N_batch, 8). Please check `StepLoader`.')
    X0, X1 = steps[:,2:5], steps[:,5:8]
    tpclabels = tpc_label(borders, X0, X1, **kwargs)
    anodes, cathodes, drift_dir = tpc_drift_direction(borders)
    unique_labels, inverse_indices = torch.unique(tpclabels, return_inverse=True, return_counts=False, sorted=True)
    tpcs = []

    for tpclabel in unique_labels:
        if tpclabel < 0:
            continue
        indices = torch.nonzero(inverse_indices == tpclabel, as_tuple=True)
        tpcs.append(TPCDataset(
            tuple(f[indices] for f in features), labels[indices], tpclabel,
            anode=anodes[tpclabel],
            cathode=cathodes[tpclabel],
            drift=drift_dir[tpclabel],
        ))
    return tpcs


class LazyLabelBatchSampler(Sampler):
    '''
    Data partition are lazily evaluated at the iterations.
    A loop on dataset is executed internally per batch.
    '''
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size

    def __iter__(self):
        current_label = None
        batch = []
        for idx, label in enumerate(self.labels):
            if current_label is None:
                current_label = label
            if label != current_label:
                for i in range(0, len(batch), self.batch_size):
                    yield batch[i:i + self.batch_size]
                batch = []
                current_label = label
            batch.append(idx)
        for i in range(0, len(batch), self.batch_size):
            yield batch[i:i + self.batch_size]
    def __len__(self):
        return len(self.__iter__())


class EagerLabelBatchSampler(Sampler):
    '''
    Data partition are eagerly evaluated at the initializations.
    '''
    def __init__(self, labels, batch_size):
        self.batch_size = batch_size
        self.batches = []

        # Precompute batches per group: each batch contains indices with the same label.
        unique_labels, inverse_indices = torch.unique(labels, sorted=True, return_inverse=True, return_counts=False)
        # Group indices by label (assuming labels are already ordered)
        for i, label in enumerate(unique_labels):
            indices = torch.nonzero(inverse_indices == i, as_tuple=True) # tuple of indices for advanced indexing
            assert len(indices) == 1
            for j in range(0, len(indices[0]), self.batch_size):
                self.batches.append(indices[0][j:j+batch_size])
    
    def __iter__(self):
        # Optionally, shuffle the batches if needed
        return iter(self.batches)


class SortedLabelBatchSampler(Sampler):
    '''
    Labels are sorted and features are reordered due to labels.
    Features are grouped by sorted labels at first eagerly.
    Lazy evaluations inside group.
    '''
    def __init__(self, labels, batch_size):
        self.batch_size = batch_size
        self.groups = []
        idxs = torch.arange(len(labels))
        unique_labels, inverse_indices = torch.unique(labels, sorted=True, return_inverse=True, return_counts=False)
        # Group indices by label (assuming labels are already ordered)
        for i, label in enumerate(unique_labels):
            indices = torch.nonzero(inverse_indices == i, as_tuple=True) # tuple of indices for advanced indexing
            assert len(indices) == 1
            self.groups.append(indices[0]) # indices along batch should be 1D
            
    def __iter__(self):
        for g in self.groups:
            for i in range(0, len(g), self.batch_size):
                yield g[i:i+self.batch_size]

    def __len__(self):
        return len(self.__iter__())

        
def nd_collate_fn(batch):
    '''
    `batch` is assumed to be (features, labels)
    NOT [(features_sample1, labels_sample1), (features_sample2, labels_sample2), ...].
    
    This function does task different from general collate_fn.
    In general, collate_fn is expected to convert [(features_sample1, labels_sample1), ...]
    to ([features_sample1, ...], [labels_sample1, ...]).
    
    This function will convert time of creation to global + offset.

    `Features` is a list/tuple, (FloatTensor, DoubleTensor, IntTensor)

    Tensors are in a shape of (N_batch, n_feature)

    The time of creation is assumed to be the first component of DoubleTensor.
    '''        
    features, labels = batch
    # features, labels = zip(*batch) # for generatel conversion from [(features_sample1, labels_sample1), ...] to ([features_sample1, ...], [labels_sample1, ...])
    t64bit = features[1] if features[1].dim() == 1 else features[1][:,0]

    # FIXME: decimals to round depends on the units. Here -3 means ms in as I assume t64bit in us.
    t = torch.min(t64bit, dim=0)[0].round(decimals=-3)
    t = t.expand(t64bit.size())
    dt = t64bit - t
    ts = torch.stack([t.to(torch.float32), dt.to(torch.float32)], dim=1)
    features = (torch.concatenate([features[0], ts], dim=1), features[1], features[2])
    
    return features, labels


class CustomNDLoader():
    '''
    Dataset must be map-style, having method __getitem__().
    '''
    def __init__(self, dataset, sampler=None, collate_fn=None, batch_size=None):
        '''
        The loader is not supposed to have automatic batching.
        
        It can only support a fixed batch size from `batch_size`, or a batch scheme provided by `sampler`.
        Arguments `sampler` and `batch_size` are mutually exclusive.
        
        Dataset must be map-style, having method __getitem__().

        Argument `collate_fn` is not in the general sense. It does not collect and combine but do transformation on the fly.
        In the original PyTorch way, collate_fn is expected to convert [(features_sample1, labels_sample1), ...]
        to ([features_sample1, ...], [labels_sample1, ...]). This function skip the step but assuming data that will be processed 
        already in the foramt of ([features_sample1, ...], [labels_sample1, ...]) and perform further transformation.

        When sampler is given, `CustomNDLoader(dataset, sampler=sampler, collate_fn=collate_fn)` is expected to behave like
        `DataLoader(dataset, sampler=sampler, collate_fn=collate_fn, batch_size=None)` (not tested).
        There is no equivalent in `DataLoader` of PyTorch for `CustomNDLoader(dataset, collate_fn=collate_fn, batch_size=16)`.
        '''
        if sampler !=None and batch_size != None:
            raise ValueError('Wrong argument combination.'
                             ' Does not support sampler and batch_size simultaneously.'
                             ' This is not like data loader in PyTorch.')
        self._dataset = dataset
        self._sampler = sampler
        if collate_fn is None:
            collate_fn = CustomNDLoader.__dummy_collate_fn
        self._collate_fn = collate_fn
        self._batch_size = batch_size

    @staticmethod
    def __dummy_collate_fn(batch):
        return batch

    def __iter__(self):
        if self._sampler is None:
            for i in range(0, len(self._dataset), self._batch_size):
                yield self._collate_fn(self._dataset[i:i+self._batch_size])
        else:
            for indices in self._sampler:
                yield self._collate_fn(self._dataset[indices])