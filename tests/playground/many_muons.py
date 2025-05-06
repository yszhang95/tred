#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import re
import os
import json

def export_event_jsons(npz_path, output_dir, prefix='effq'):
    f = np.load(npz_path)

    # Match pattern only for exact keys like 'effq_tpcX_batchY'
    pattern = re.compile(rf'{prefix}_tpc(\d+)_batch(\d+)$')
    matched_keys = [k for k in f.files if pattern.match(k)]

    all_positions = []
    all_charges = []
    all_event_ids = []
    all_tpc_ids = []

    for key in matched_keys:
        match = pattern.match(key)
        tpc = int(match.group(1))
        batch = int(match.group(2))

        data = f[key]
        num_hits = data.shape[0]
        event_id = int(f[f'event_id_tpc{tpc}_batch{batch}'])
        event_ids = np.full(num_hits, event_id, dtype=int)
        tpc_ids = np.full(num_hits, tpc, dtype=int)

        all_positions.append(data[:, :3])
        all_charges.append(data[:, -1])
        all_event_ids.append(event_ids)
        all_tpc_ids.append(tpc_ids)

    # Concatenate all arrays
    positions = np.vstack(all_positions)
    charges = np.hstack(all_charges)
    event_ids = np.hstack(all_event_ids)
    tpc_ids = np.hstack(all_tpc_ids)

    # Sort by event_id
    sort_indices = np.argsort(event_ids)
    positions = positions[sort_indices]
    charges = charges[sort_indices]
    event_ids_sorted = event_ids[sort_indices]
    tpc_ids = tpc_ids[sort_indices]

    # Grouping by event_id
    _, idx_split = np.unique(event_ids_sorted, return_index=True)
    pos_groups = np.split(positions, idx_split[1:])
    charge_groups = np.split(charges, idx_split[1:])
    event_ids_unique = event_ids_sorted[idx_split]
    tpc_groups = np.split(tpc_ids, idx_split[1:])

    # Write each group to JSON
    for eid, pos, chg, tpcs in zip(event_ids_unique, pos_groups, charge_groups, tpc_groups):
        evt = {
            "runNo": 0,
            "subRunNo": 0,
            "eventNo": int(eid),
            "x": pos[:, 0].tolist(),
            "y": pos[:, 1].tolist(),
            "z": pos[:, 2].tolist(),
            "q": chg.tolist(),
            "nq": [0] * len(chg),
            "cluster_id": tpcs.tolist(),
            "real_cluster_id": tpcs.tolist(),
            "geom": "uboone",
            "type": "wire-cell",
        }

        # Save to subfolder per event
        edir = os.path.join(output_dir, str(eid))
        os.makedirs(edir, exist_ok=True)
        filename = f"{eid}-{prefix}.json"
        with open(os.path.join(edir, filename), "w") as fjson:
            json.dump(evt, fjson, indent=2)

    print(f"Exported {len(event_ids_unique)} event JSONs to '{output_dir}'")

# Example usage:
# export_event_jsons("output_hits.npz", "mu_jsons/data", prefix="effq")
# export_event_jsons("output_hits.npz", "mu_jsons/data", prefix="hits")


# In[2]:


export_event_jsons("output_hits.npz", "mu_jsons/data", prefix="effq")


# In[3]:


export_event_jsons("output_hits.npz", "mu_jsons/data", prefix="hits")


# In[4]:


import numpy as np
import re
import json
import os
from collections import defaultdict

def export_grouped_mc_segments(npz_path, output_base_dir):
    f = np.load(npz_path)

    # Pattern to match event_id keys
    pattern = re.compile(r'event_id_tpc(\d+)_batch(\d+)$')
    event_keys = [k for k in f.files if pattern.match(k)]

    grouped = defaultdict(list)

    for k in event_keys:
        match = pattern.match(k)
        tpc = int(match.group(1))
        batch = int(match.group(2))

        eid   = int(f[f'event_id_tpc{tpc}_batch{batch}'])
        start = f[f'event_start_tpc{tpc}_batch{batch}'].tolist()
        end   = f[f'event_end_tpc{tpc}_batch{batch}'].tolist()

        segment = {
            "id": int(f"{eid}{tpc:02d}{batch:05d}"),  # Unique ID per TPC
            "text": f"muon TPC{tpc} batch{batch}",
            "data": {
                "start": start,
                "end": end,
            },
            "children": [],
            "icon": "jstree-file"
        }

        grouped[eid].append(segment)

    # Write one JSON per event ID
    for eid, segments in grouped.items():
        outdir = os.path.join(output_base_dir, str(eid))
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f"{eid}-mc.json")
        with open(outpath, "w") as fjson:
            json.dump(segments, fjson, indent=2)

    print(f"Exported {len(grouped)} event MC JSONs to '{output_base_dir}'")

# Example usage:
# export_grouped_mc_segments("output_hits.npz", "mu_jsons/data")


# In[5]:


export_grouped_mc_segments("output_hits.npz", "mu_jsons/data")


# In[6]:


import numpy as np
import re
import os
import json

def export_event_array(npz_path, output_dir, prefix='effq'):
    f = np.load(npz_path)

    # Match pattern only for exact keys like 'effq_tpcX_batchY'
    pattern = re.compile(rf'{prefix}_tpc(\d+)_batch(\d+)$')
    matched_keys = [k for k in f.files if pattern.match(k)]

    all_positions = []
    all_charges = []
    all_event_ids = []
    all_tpc_ids = []
    all_indices = []

    for key in matched_keys:
        match = pattern.match(key)
        tpc = int(match.group(1))
        batch = int(match.group(2))

        data = f[key]
        num_hits = data.shape[0]
        event_id = int(f[f'event_id_tpc{tpc}_batch{batch}'])
        event_ids = np.full(num_hits, event_id, dtype=int)
        tpc_ids = np.full(num_hits, tpc, dtype=int)

        all_positions.append(data[:, :3])
        all_charges.append(data[:, -1])
        all_event_ids.append(event_ids)
        all_tpc_ids.append(tpc_ids)
        all_indices.append(f[f'{key}_location'])

    # Concatenate all arrays
    positions = np.vstack(all_positions)
    charges = np.hstack(all_charges)
    event_ids = np.hstack(all_event_ids)
    tpc_ids = np.hstack(all_tpc_ids)
    indices = np.vstack(all_indices)

    # Sort by event_id
    sort_indices = np.argsort(event_ids)
    positions = positions[sort_indices]
    charges = charges[sort_indices]
    event_ids = event_ids[sort_indices]
    tpc_ids = tpc_ids[sort_indices]
    indices = indices[sort_indices]
    return positions, charges, event_ids, tpc_ids, indices


# In[7]:


p, c, e, t, i = export_event_array("output_hits.npz", "mu_jsons/data", prefix="hits")


# In[8]:


i[-1], i[-2], e[-1], e[-2]


# In[9]:


import h5py


# In[10]:


with h5py.File('many_muons.hdf5', 'w') as fout:
    for k in ['hits', 'effq']:
        arrs = export_event_array("output_hits.npz", "mu_jsons/data", prefix=k)
        g = fout.create_group(k)
        g.create_dataset('position', data=arrs[0])
        g.create_dataset('charge', data=arrs[1])
        g.create_dataset('event_id', data=arrs[2])
        g.create_dataset('tpc_id', data=arrs[3])
        g.create_dataset('grid_index', data=arrs[4])


# In[18]:


for i in range(0, 70):
    print('tpc index', i, 'drift direction', np.load('output_hits.npz')[f'drift_direction_tpc{i}'], 'is index even?', i % 2)


# In[ ]:




