#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
import matplotlib.cm as cm

import json


# In[2]:


f = np.load('waveforms.npz')
print(f.files)
print('yz 27', f['tpc_lower_left_tpc27'])
print('yz 29', f['tpc_lower_left_tpc29'])

positions = {
    'tpc27': None,
    'tpc29': None
}
charges = {
    'tpc27' : None,
    'tpc29' : None
}
for i in [27, 29]:
    locs = f[f'effq_tpc{i}_batch0'][:,:3]
    drift_direction = f[f'drift_direction_tpc{i}']
    tpc_anode = f[f'tpc_anode_{i}']

    print('lower left', f[f'tpc_lower_left_tpc{i}'])
    print(f'direction {i}', drift_direction)
    print(f'anode {i}', tpc_anode)

    positions[f'tpc{i}'] = locs
    charges[f'tpc{i}'] = f[f'effq_tpc{i}_batch0'][:,-1]
    print(f[f'global_tref_tpc{i}_batch0'])
    print(f[f'event_start_tpc{i}_batch0'])
    print(f[f'event_end_tpc{i}_batch0'])
    print(f[f'event_id_tpc{i}_batch0'])

# In[3]:

evt = {
    "runNo"    : "10000007",
    "subRunNo" : "16",
    "eventNo"  : "1501",
    "x"        : [],
    "y"        : [],
    "z"        : [],
    "q"        : [],
    "nq"       : [],
    "cluster_id" : [],
    "real_cluster_id" : [],
    "geom"     : "uboone",
    "type"     : "wire-cell",
}

for i in [27, 29]:
    evt['x'] += positions[f'tpc{i}'][:,0].tolist()
    evt['y'] += positions[f'tpc{i}'][:,1].tolist()
    evt['z'] += positions[f'tpc{i}'][:,2].tolist()
    evt['q'] += charges[f'tpc{i}'].tolist()
    evt['cluster_id'] += [i,] * len(evt['x'])
    evt['real_cluster_id'] += [i,] * len(evt['x'])
evt['nq'] = [0,] * len(evt['x'])

# Write to a JSON file
with open("myfile/data/0/0-mueffq.json", "w") as fjson:
    json.dump(evt, fjson, indent=2)  # `indent` makes it pretty-printed

for i in [27, 29]:
    locs = f[f'hits_tpc{i}_batch0'][:,:3]
    drift_direction = f[f'drift_direction_tpc{i}']
    tpc_anode = f[f'tpc_anode_{i}']
    positions[f'tpc{i}'] = locs
    charges[f'tpc{i}'] = f[f'hits_tpc{i}_batch0'][:,-1]

evthits = {
    "runNo"    : "10000007",
    "subRunNo" : "16",
    "eventNo"  : "1501",
    "x"        : [],
    "y"        : [],
    "z"        : [],
    "q"        : [],
    "nq"       : [],
    "cluster_id" : [],
    "real_cluster_id" : [],
    "geom"     : "uboone",
    "type"     : "wire-cell",
}

for i in [27, 29]:
    evthits['x'] += positions[f'tpc{i}'][:,0].tolist()
    evthits['y'] += positions[f'tpc{i}'][:,1].tolist()
    evthits['z'] += positions[f'tpc{i}'][:,2].tolist()
    evthits['q'] += charges[f'tpc{i}'].tolist()
    evthits['cluster_id'] += [i,] * len(evthits['x'])
    evthits['real_cluster_id'] += [i,] * len(evthits['x'])
evthits['nq'] = [0,] * len(evthits['x'])

# Write to a JSON file
with open("myfile/data/0/0-muhits.json", "w") as fjson:
    json.dump(evthits, fjson, indent=2)  # `indent` makes it pretty-printed


musegs = []

for i in [27, 29]:
    mucopy = {
        "id": int(f"{f[f'event_id_tpc{i}_batch0']}{i:02d}"),
        "text": f"muon TPC{i}",
        "data": {
            "start": f[f'event_start_tpc{i}_batch0'].tolist(),
            "end": f[f'event_end_tpc{i}_batch0'].tolist(),
        },
        "children": [],
        "icon": "jstree-file"
    }
    musegs.append(mucopy)
print(musegs)
with open("myfile/data/0/0-mc.json", "w") as fjson:
    json.dump(musegs, fjson, indent=2)  # `indent` makes it pretty-printed


charge_color = np.array(evt['q'])

plt.plot(positions['tpc27'][:,0], positions['tpc27'][:,2], 'o', label='tpc27')
plt.plot(positions['tpc29'][:,0], positions['tpc29'][:,2], 'o', label='tpc29')
plt.xlabel('x [cm]')
plt.ylabel('z [cm]')
plt.title('from hits')
plt.legend()

plt.plot(positions['tpc27'][:,0], positions['tpc27'][:,1], 'o', label='tpc27')
plt.plot(positions['tpc29'][:,0], positions['tpc29'][:,1], 'o', label='tpc29')
plt.xlabel('x [cm]')
plt.ylabel('y [cm]')
plt.title('from hits')
plt.legend()

plt.plot(positions['tpc27'][:,2], positions['tpc27'][:,1], 'o', label='tpc27')
plt.plot(positions['tpc29'][:,2], positions['tpc29'][:,1], 'o', label='tpc29')
plt.xlabel('z [cm]')
plt.ylabel('y [cm]')
plt.title('from hits')
plt.legend()

import os
import zipfile

base_dir = "myfile/data"  # relative to current dir "A"
zip_filename = "myfile/myfile.zip"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)
            # This will ensure the files are placed under C/ when unzipped
            arcname = os.path.relpath(full_path, os.path.dirname(base_dir))
            zipf.write(full_path, arcname)
