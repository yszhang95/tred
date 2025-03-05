#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import h5py
import numpy as np
import yaml
from collections import defaultdict
import torch
import time

from torch.utils.data import DataLoader


# In[2]:


import sys
sys.path.append('/home/yousen/Documents/NDLAr2x2/tred/src')
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
    LazyLabelBatchSampler, EagerLabelBatchSampler, SortedLabelBatchSampler, CustomNDLoader, simple_geo_parser
)


# In[3]:


borders = simple_geo_parser('../nd_geometry/ndlar-module.yaml', '../nd_geometry/multi_tile_layout-3.0.40.yaml')


# In[4]:


path = '/home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5'
d0 = StepLoader(h5py.File(path), transform=steps_from_ndh5)


# In[5]:


f0, f1, i0 = d0[:]
tpcs = create_tpc_datasets_from_steps((f0, f1, i0), i0, borders, sort_index=1)


# In[6]:


def is_correct(tpcdataset, use_collate=False):
    label_index = 0
    d0 = {}
    d1 = {}
    collate_fn = nd_collate_fn if use_collate else None
    for btype, sampler in zip(
        ['lazy', 'sorted', 'eager'], [LazyLabelBatchSampler, SortedLabelBatchSampler, EagerLabelBatchSampler]
    ):
        dl0 = DataLoader(tpcdataset, sampler=sampler(tpcdataset.labels[:,label_index], 4096), batch_size=None, collate_fn=collate_fn)
        dl1 = CustomNDLoader(tpcdataset, sampler=sampler(tpcdataset.labels[:,label_index], 4096), batch_size=None, collate_fn=collate_fn)
        d0[btype] = [d for d in dl0]
        d1[btype] = [d for d in dl1]

    for btype in d0.keys():
        assert len(d0[btype]) == len(d0[btype])
        for b in d0[btype]:
            ls = torch.unique(b[1][:,label_index])
            assert len(ls) == 1
        l = torch.concat([d[1] for d in d0[btype]], dim=0)
        # print(btype, len(tpcdataset.labels.equal(torch.concat([d[1] for d in d0[btype]], dim=0))))
        assert tpcdataset.labels.equal(torch.concat([d[1] for d in d0[btype]], dim=0))
        assert tpcdataset.labels.equal(torch.concat([d[1] for d in d1[btype]], dim=0))

        for i in range(3):
            s = (slice(None,None,None), slice(0,8,None)) if i == 0 else slice(None,None,None)
            assert tpcdataset.features[i].allclose(torch.concat([d[0][i][s] for d in d0[btype]], dim=0))
            assert tpcdataset.features[i].allclose(torch.concat([d[0][i][s] for d in d1[btype]], dim=0))
    if use_collate:
        for i in range(3):
            print(d0[btype][i][0][0][:3,8:], 'sum', torch.sum(d0[btype][i][0][0][:3,8:].to(torch.float64), dim=1), d0[btype][i][0][1][:3])


# In[7]:


is_correct(tpcs[1])
is_correct(tpcs[1], True)


# In[8]:


stime = time.time()
loader = DataLoader(tpcs[0], sampler=LazyLabelBatchSampler(tpcs[0].labels[:,1], 4096), batch_size=None)
nelements = 0
for batch_idx, (features, label_batch) in enumerate(loader):
    nelements += len(features[0])
etime = time.time()
print(etime - stime)


# In[9]:


stime = time.time()
loader = CustomNDLoader(tpcs[0], sampler=LazyLabelBatchSampler(tpcs[0].labels[:,1], 4096), batch_size=None)
nelements = 0
for batch_idx, (features, label_batch) in enumerate(loader):
    nelements += len(features[0])
etime = time.time()
print(nelements)
print(etime - stime)


# In[10]:


stime = time.time()
batch_sampler = SortedLabelBatchSampler(tpcs[0].labels[:,1], 4096)
loader = CustomNDLoader(tpcs[0], sampler=batch_sampler)
nelements = 0
for batch_idx, (features, label_batch) in enumerate(loader):
    nelements += len(features[0])
etime = time.time()
print(nelements)
print(etime - stime)


# In[11]:


for itpc in range(len(tpcs)):
    loader = CustomNDLoader(tpcs[itpc], sampler=LazyLabelBatchSampler(tpcs[itpc].labels[:,1], 4096), batch_size=None)
    nelements = 0
    stime = time.time()
    for batch_idx, (features, label_batch) in enumerate(loader):
        nelements += len(features[0])
    etime = time.time()
    print(itpc, etime - stime, nelements, (etime-stime)/nelements * 1E9)


# In[12]:


for itpc in range(len(tpcs)):
    loader = CustomNDLoader(tpcs[itpc], sampler=SortedLabelBatchSampler(tpcs[itpc].labels[:,1], 4096), batch_size=None)
    nelements = 0
    stime = time.time()
    for batch_idx, (features, label_batch) in enumerate(loader):
        nelements += len(features[0])
    etime = time.time()
    print(itpc, etime - stime, nelements, 'average time per elements in the batch (ns)', (etime-stime)/nelements * 1E9)


# In[13]:


stime = time.time()
loader = CustomNDLoader(tpcs[0], sampler=EagerLabelBatchSampler(tpcs[0].labels[:,1], 4096), batch_size=None)
nelements = 0
for batch_idx, (features, label_batch) in enumerate(loader):
    nelements += len(features[0])
etime = time.time()
print(nelements)
print(etime - stime)


# In[14]:


for itpc in range(len(tpcs)):
    loader = CustomNDLoader(tpcs[itpc], sampler=EagerLabelBatchSampler(tpcs[itpc].labels[:,1], 4096), batch_size=None)
    nelements = 0
    stime = time.time()
    for batch_idx, (features, label_batch) in enumerate(loader):
        nelements += len(features[0])
    etime = time.time()
    print(itpc, etime - stime, nelements, (etime-stime)/nelements * 1E9)


# In[15]:


for itpc in range(len(tpcs)):
    loader = DataLoader(tpcs[itpc], sampler=EagerLabelBatchSampler(tpcs[itpc].labels[:,1], 4096), batch_size=None)
    nelements = 0
    stime = time.time()
    for batch_idx, (features, label_batch) in enumerate(loader):
        nelements += len(features[0])
    etime = time.time()
    print(itpc, etime - stime, nelements, (etime-stime)/nelements * 1E9)


# In[16]:


for itpc in range(len(tpcs)):
    loader = CustomNDLoader(tpcs[itpc], sampler=SortedLabelBatchSampler(tpcs[itpc].labels[:,1], 4096), batch_size=None)
    nelements = 0
    stime = time.time()
    for batch_idx, (features, label_batch) in enumerate(loader):
        nelements += len(features[0])
    etime = time.time()
    print(itpc, etime - stime, nelements, (etime-stime)/nelements * 1E9)


# In[17]:


for itpc in range(len(tpcs)):
    loader = CustomNDLoader(tpcs[itpc], sampler=SortedLabelBatchSampler(tpcs[itpc].labels[:,1], 4096), batch_size=None,
                           collate_fn=nd_collate_fn)
    nelements = 0
    stime = time.time()
    for batch_idx, (features, label_batch) in enumerate(loader):
        nelements += len(features[0])
    etime = time.time()
    print(itpc, etime - stime, nelements, (etime-stime)/nelements * 1E9)


# In[18]:


for itpc in range(len(tpcs)):
    loader = CustomNDLoader(tpcs[itpc], sampler=None, batch_size=4096)
    nelements = 0
    stime = time.time()
    for batch_idx, (features, label_batch) in enumerate(loader):
        nelements += len(features[0])
    etime = time.time()
    print(itpc, etime - stime, nelements, (etime-stime)/nelements * 1E9)


# In[19]:


for itpc in range(len(tpcs)):
    loader = CustomNDLoader(tpcs[itpc], sampler=None, batch_size=4096)
    nelements = 0
    stime = time.time()
    for batch_idx, (features, label_batch) in enumerate(loader):
        nelements += len(features[0])
    etime = time.time()
    print(itpc, etime - stime, nelements, (etime-stime)/nelements * 1E9)


# In[ ]:




