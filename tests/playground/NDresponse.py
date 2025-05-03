#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import requests
import os


# ## Download the response file

# In[2]:


url = 'https://raw.githubusercontent.com/DUNE/larnd-sim/develop/larndsim/bin/response_44_v2a_full.npz'
filename = os.path.basename(url)

if not os.path.isfile(filename):
    print(f"Downloading {filename} ...")
    response = requests.get(url)
    response.raise_for_status()  # raise an error if the download failed
    with open(filename, 'wb') as f:
        f.write(response.content)
    print("Download complete.")
else:
    print(f"{filename} already exists. Skipping download.")


# ## Content of the file

# In[3]:


f = np.load('response_44_v2a_full.npz')
response = f['response']
print('Keys in npz', f.files)
print('response array shape', response.shape)
print('drift length', f['drift_length'], 'cm')
print('time_tick', f['time_tick'], 'us')
print('bin_size', f['bin_size'], 'cm')


# ## Symmetry in the response array

# ### Rotational symmetry by studying orthogonal axes
# I expect the rotation symmetry every 90 degrees.

# #### In the precision of 50ns per time tick

# In[4]:


fig, axes = plt.subplots(1,2, figsize=(6*2, 6))
for i in range(5):
    for j in range(1):
        axes[0].plot(response[j,i]-response[i,j], label=f'response [{j},{i}]-[{i},{j}]')
        axes[1].plot(response[j,i], label=f'response [{j},{i}]')
axes[0].set_xlim(3780, 3850)
axes[1].set_xlim(3780, 3850)
axes[0].set_xlabel('index along time')
axes[1].set_xlabel('index along time')
axes[0].set_ylabel('response[j,i]-response[i,j]')
axes[1].set_ylabel('response[j,i]')
axes[0].legend(loc='upper right')
axes[1].legend(loc='upper right')


# #### In the precision of 500ns per time tick (by summing original array every 10 bins)

# In[5]:


fig, axes = plt.subplots(1,2, figsize=(6*2, 6))
for i in range(5):
    for j in range(1):
        axes[0].plot(response[j,i].reshape(-1,10).sum(axis=1) - response[i,j].reshape(-1,10).sum(axis=1), label=f'response [{j},{i}]-[{i},{j}]; sum 10 bins')
        axes[1].plot(response[j,i].reshape(-1,10).sum(axis=1), label=f'response [{j},{i}]; sum 10 bins')
axes[0].set_xlim(374, 385)
axes[1].set_xlim(374, 385)
axes[0].legend(loc='upper right')
axes[1].legend(loc='upper right')
axes[0].set_xlabel('index along time')
axes[1].set_xlabel('index along time')
axes[0].set_ylabel('response[j,i]-response[i,j]')
axes[1].set_ylabel('response[j,i]')


# ### Reflection symmetry over a diagonal

# In[6]:


fig, axes = plt.subplots(1,2, figsize=(6*2, 6))
for i in range(1,5):
    for j in range(i+1,5):
        axes[0].plot(response[j,i]-response[i,j], label=f'response [{j},{i}]-[{i},{j}]')
        axes[1].plot(response[j,i], label=f'response [{j},{i}]')
axes[0].set_xlim(3780, 3850)
axes[1].set_xlim(3780, 3850)
axes[0].legend(loc='upper right')
axes[1].legend(loc='upper right')
axes[0].set_xlabel('index along time')
axes[1].set_xlabel('index along time')
axes[0].set_ylabel('response[j,i]-response[i,j]')
axes[1].set_ylabel('response[j,i]')


# In[8]:


f.files


# In[25]:


start = 2500
distance = f['drift_length'] - start*f['time_tick']*0.16 # cm/us * us
print(f['bin_size'])


# In[26]:


ofile = f'response_v2a_distance_{distance:.3f}cm_binsize_{f["bin_size"]}cm_tick{f["time_tick"]}us'.replace('.', 'p')
np.save(f"{ofile}.npy", response[:,:,start:])
