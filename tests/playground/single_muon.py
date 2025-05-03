#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
import matplotlib.cm as cm

import json

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
    location = f[f'hits_tpc{i}_batch0_location'][:,[2,0,1]]
    locs = np.zeros_like(location, dtype=float)
    # print(np.unique(location[:,1:3], axis=0))
    locs[:,[1,2]] = location[:,[1,2]].astype(float)*0.44 + f[f'tpc_lower_left_tpc{i}']
    drift_direction = f[f'drift_direction_tpc{i}']
    tpc_anode = f[f'tpc_anode_{i}']

    print('lower left', f[f'tpc_lower_left_tpc{i}'])
    print(f'direction {i}', drift_direction)
    print(f'anode {i}', tpc_anode)

    locs[:,0] = 50.4 + tpc_anode - drift_direction * 0.16 * 0.05 * location[:,0].astype(float)
    positions[f'tpc{i}'] = locs
    charges[f'tpc{i}'] = f[f'hits_tpc{i}_batch0']

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
with open("myfile/data/0/0-mu.json", "w") as f:
    json.dump(evt, f, indent=2)  # `indent` makes it pretty-printed

mu = {"id":63006,"text":"proton 118 MeV","data":{"start":[132.6, 29.6, 763.5],"end":[125.3, 22.1, 766.2]},"children":[],"icon":"jstree-file"}
fmu = h5py.File('/home/yousen/Public/ndlar_shared/data/segments_a_muon_first_event.hdf5')
musegs = []
start_coord = np.array([fmu['segments'][f'{i}_start'].astype(float) for i in ['x', 'y', 'z']]).T
end_coord = np.array([fmu['segments'][f'{i}_end'].astype(float) for i in ['x', 'y', 'z']]).T

mu['text'] = 'muon'
mu['data']['start'] = start_coord[0].tolist()
mu['data']['end'] = end_coord[-1].tolist()

musegs.append(
    mu
)
with open("myfile/data/0/0-mc.json", "w") as f:
    json.dump(musegs, f, indent=2)  # `indent` makes it pretty-printed

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
