#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
import matplotlib.cm as cm

import k3d


# In[ ]:


f = np.load('waveforms.npz')
print(f.files)
location = f['current_tpc27_batch0_location']
data = f['current_tpc27_batch0_data']

location = f['effq_tpc27_batch0_location']
data = f['effq_tpc27_batch0_data']

location = f['effq_tpc29_batch0_location']
data = f['effq_tpc29_batch0_data']

Nbatch, Nx, Ny, Nz = data.shape
print(Nbatch)

data = data.reshape(Nbatch, Nx//10, 10, Ny//10, 10, Nz//10, 10)

data = data.transpose((0, 1, 3, 5, 2, 4, 6))
data = np.sum(data, axis=(4,5,6))

Nbatch, Nx, Ny, Nz = data.shape


# In[ ]:


# Grid spacing
spacing = (0.038, 0.038, 0.05*0.16) # (cm, cm, cm)

# Collect all filtered points
all_X, all_Y, all_Z, all_V = [], [], [], []
print(Nbatch)
for b in range(Nbatch):
    origin = location[b]
    # print(origin)
    x = np.arange(Nx) * spacing[0] + origin[0] * spacing[0]
    y = np.arange(Ny) * spacing[1] + origin[1] * spacing[1]
    z = np.arange(Nz) * spacing[2] + origin[2] * spacing[2]

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V = data[b]

    # Flatten
    Xf, Yf, Zf, Vf = X.flatten(), Y.flatten(), Z.flatten(), V.flatten()

    # Apply mask
    mask = Vf >= -10
    all_X.append(Xf[mask])
    all_Y.append(Yf[mask])
    all_Z.append(Zf[mask])
    all_V.append(Vf[mask])

# Combine all batches
X_all = np.concatenate(all_X)
Y_all = np.concatenate(all_Y)
Z_all = np.concatenate(all_Z)
V_all = np.concatenate(all_V)

print(len(V_all))

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_all, Y_all, Z_all, c=V_all, cmap='viridis', alpha=0.7)
plt.colorbar(sc, label='Value')
ax.set_title("Combined Grid Blocks (Values ? 1)")
plt.show()

fig, ax = plt.subplots()
ax.scatter(X_all, Y_all)


# In[ ]:


# Normalize V_all to use as color
vmin, vmax = np.min(V_all), np.max(V_all)
V_norm = (V_all - vmin) / (vmax - vmin)

# Choose a colormap (here, 'viridis' converted to RGB using matplotlib)
# Convert normalized values to RGB
rgb = cm.viridis(V_norm)[:, :3]  # Drop alpha channel

# Convert to 0xRRGGBB format expected by k3d (uint32)
colors = (rgb * 255).astype(np.uint8)
colors_uint32 = (colors[:, 0].astype(np.uint32) << 16) + \
                (colors[:, 1].astype(np.uint32) << 8) + \
                (colors[:, 2].astype(np.uint32))
# Create plot
plot = k3d.plot()

points = k3d.points(
    positions=np.vstack([X_all, Y_all, Z_all]).T.astype(np.float32),
    colors=colors_uint32,
    point_size=0.5,
    shader='3d'
)
display(plot)
plot += points

