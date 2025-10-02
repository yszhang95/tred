# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% id="tA5KNieJ9nJg"
import matplotlib.pyplot as plt
import numpy as np
import torch

# %% id="XGdduSfOSt_i"
import tred
from tred.blocking import Block
from tred.chunking import accumulate
from tred.sparse import chunkify, SGrid

# %%
import plotly.graph_objects as go

# %% id="LfGqRreb9p3E"
import torch
from torch import nn

import torch
import torch.nn.functional as F

def gaussian_kernel1d(kernel_size: int, sigma: float, device=None):
    x = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

def smooth_along_last_dim(x, kernel, padding="same"):
    """
    Apply 1D convolution with given kernel along the last dimension of x.
    Works for arbitrary-shaped tensors.
    """
    original_shape = x.shape
    L = original_shape[-1]
    batch = int(torch.tensor(original_shape[:-1]).prod())  # flatten all but last
    x_reshaped = x.reshape(batch, 1, L)                   # (batch, channels=1, length)

    kernel = kernel.view(1, 1, -1)  # (out_channels=1, in_channels=1, kernel_size)

    y = F.conv1d(x_reshaped, kernel, padding=padding)

    return y.reshape(*original_shape[:-1], y.shape[-1])   # back to original shape

def smooth_gaussian(x, kernel_size=7, sigma=1.0):
    kernel = gaussian_kernel1d(kernel_size, sigma, device=x.device)
    return smooth_along_last_dim(x, kernel)

def smooth_uniform(x, kernel_size=7):
    kernel = torch.ones(kernel_size, device=x.device) / kernel_size
    return smooth_along_last_dim(x, kernel)


# ---------- FFT-based 3D linear convolution ----------
def fft_convolve3d(x, k):
    """
    Linear 3D convolution via FFT.

    x: (1, D, H, W)
    k: (Kd, Kh, Kw) or broadcastable to x (no channel dim assumed)
    mode: 'same' -> output (1, D, H, W), 'full' -> (1, D+Kd-1, H+Kh-1, W+Kw-1)
    """
    assert x.dim() == 4 and x.shape[0] == 1, f"x must be (1, D, H, W), x.shape=={x.shape}"
    D, H, W = x.shape[1:]
    Kd, Kh, Kw = k.shape[-3:]

    # Full linear conv size
    Df, Hf, Wf = D + Kd - 1, H + Kh - 1, W + Kw - 1

    # Next pow2 sizes for speed
    # def next_pow2(n):
    #     return 1 << (n - 1).bit_length()
    # nD, nH, nW = next_pow2(Df), next_pow2(Hf), next_pow2(Wf)
    nD, nH, nW = Df, Hf, Wf

    # FFTs
    Xf = torch.fft.rfftn(x, s=(nD, nH, nW), dim=(1,2,3))
    Kf = torch.fft.rfftn(k, s=(nD, nH, nW), dim=(0,1,2))
    Yf = Xf * Kf  # broadcast over batch

    y_full = torch.fft.irfftn(Yf, s=(nD, nH, nW), dim=(1,2,3))[:, :Df, :Hf, :]

    y_center = y_full[:, Kd//2:D+Kd//2, Kh//2:H+Kh//2, :]

    return y_center

# ---------- First difference along the last axis ----------
def diff_3d(x):
    """First difference along the last axis (time)."""
    dx = x[:, 1:, :, :] - x[:, :-1, :, :]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dz = x[:, :, :, 1:] - x[:, :, :, :-1]
    return dx, dy, dz

# ---------- Apply a matrix along the last axis ----------
def apply_mat_last_axis(t, M):
    """
    t: (..., W) volume or tensor
    M: (Mout, W) matrix acting on the last axis
    returns: (..., Mout)
    """
    *lead, W = t.shape
    t2 = t.reshape(-1, W)                # (N, W)
    out = t2 @ M                       # (N, Mout)
    return out.reshape(*lead, M.shape[0])

# ---------- Objective module ----------
class Deconv3DObjective(nn.Module):
    def __init__(self, Ae, A0, K, Mask, smooth_kernel, lam_l1=0., lam_dx=0., lam_a0=0.):
        super().__init__()
        # store as buffers so they follow .to(device)
        self.register_buffer("Ae", Ae)    # (M, W')
        self.register_buffer("A0", A0)    # (P, W')
        self.register_buffer("K",  K)     # (Kd, Kh, Kw)
        self.lam_l1 = lam_l1
        self.lam_dx = lam_dx
        self.lam_a0 = lam_a0
        self.register_buffer("Mask", Mask)
        self.relu = nn.ReLU()
        self.register_buffer("smooth_kernel", smooth_kernel)

    def forward(self, Z, Y):
        """
        Z: (1, D, H, W) unconstrained
        Y: (1, D', H', M) with M == Ae.shape[0]
        """
        X = self.relu(Z) * self.Mask                             # enforce X >= 0
        X = smooth_along_last_dim(X, self.smooth_kernel)
        KX = fft_convolve3d(X, self.K)   # (1, D', H', W')
        # print('Input to FFT', X.shape, 'Kernel to FFT', self.K.shape,
        #       'KX after FFT', KX.shape, self.Ae.shape, self.A0.shape)

        # Apply Ae and A0 along the last axis independently at each (d,h) location
        # AeKX = apply_mat_last_axis(KX, self.Ae)      # (1, D', H', M)
        # A0KX = apply_mat_last_axis(KX, self.A0)      # (1, D', H', P)
        # print(self.Ae.dtype, KX.dtype)
        AeKX = self.Ae.to(KX.dtype) @ KX.unsqueeze(-1)
        AeKX = AeKX.squeeze(-1)
        A0KX = self.A0.to(KX.dtype) @ KX.unsqueeze(-1)
        A0KX = A0KX.squeeze(-1)

        # Data fidelity
        data_term = torch.sum((AeKX - Y) ** 2)

        # Regularizers
        l1_term = self.lam_l1 * torch.sum(X)         # X >= 0 -> |X| = X
        # l1_term = self.lam_l1 * torch.sum(X**2) # it is l2 term
        dx, dy, dz = diff_3d(X)
        dx_term = self.lam_dx * torch.sqrt((dx**2).sum() + (dy**2).sum() + (dz**2).sum())
        a0_term = self.lam_a0 * torch.linalg.vector_norm(A0KX)

        loss = data_term + l1_term + dx_term + a0_term
        return loss, {"X": X, "KX": KX, "AeKX": AeKX, "A0KX": A0KX,
                      "data": data_term, "l1": l1_term, "dx": dx_term, "a0": a0_term}

# ---------- Solver ----------
def solve_nonnegative_3d(
    Ae, A0, K, Y, Mask, smooth_kernel,
    lam_l1=0., lam_dx=0., lam_a0=0.,
    steps=1000, lr=1e-2, Z0=None, device=None, progress_every=100
):
    """
    Ae: (1, D', H', M, W')  matrix on last axis
    A0: (1, D', H', P, W')  matrix on last axis (cumulative integrator)
    K : (Kd, Kh, Kw)
    Y : (1, D', H', M)  matches Ae applied to last axis
    Mask : Mask on X
    Returns: M @ X_hat (1, D, H, W) and a small info dict
    """
    dev = device or (Ae.device if isinstance(Ae, torch.Tensor) else "cpu")
    Ae, A0, K, Y, Mask = Ae.to(dev), A0.to(dev), K.to(dev), Y.to(dev), Mask.to(dev)

    # Infer input spatial size if using "same": W' == input W
    # We'll initialize Z to match the *pre-conv* size implied by 'same' mode.

    BAe, Dp, Hp, M, Wp = Ae.shape
    assert BAe == 1 and Dp == Y.shape[1] and Hp == Y.shape[2] and M == Y.shape[3], f'Ae.shape {Ae.shape} does not match on Y.shape {Y.shape}'
    B, _, _, _ = Y.shape
    assert B == 1, 'batch size != 1'

    # Input W must equal Ae's width W' under 'same' mode
    Kd, Kh, Kw = K.shape
    W = Wp - Kw + 1
    D = Dp
    H = Hp
    assert all(d1 == d2 for d1, d2 in zip(Mask.shape[1:], (D, H, W))), f"Mask.shape[1:] {Mask.shape[1:]}, (D,H,W) {(D,H,W)}"
    init_shape = (1, D, H, W)

    # Initialize Z (unconstrained)
    if Z0 is None:
        Z = torch.full(init_shape, -2.0, device=dev, requires_grad=True)
    else:
        Z = Z0.clone().detach().to(dev).requires_grad_(True)

    obj = Deconv3DObjective(Ae, A0, K, Mask, smooth_kernel=smooth_kernel,
                            lam_l1=lam_l1, lam_dx=lam_dx, lam_a0=lam_a0).to(dev)
    opt = torch.optim.Adam([Z], lr=lr)

    history = []
    for t in range(1, steps + 1):
        opt.zero_grad()
        loss, parts = obj(Z, Y)
        loss.backward()
        opt.step()

        if progress_every and (t % progress_every == 0 or t == 1 or t == steps):
            with torch.no_grad():
                history.append({
                    "step": t,
                    "loss": float(loss),
                    "data": float(parts["data"]),
                    "l1": float(parts["l1"]),
                    "dx": float(parts["dx"]),
                    "a0": float(parts["a0"]),
                })

    with torch.no_grad():
        X_hat = smooth_along_last_dim(obj.relu(Z)*Mask, smooth_kernel)
        KX    = fft_convolve3d(X_hat, K)
        AeKX  = (Ae.to(KX.dtype) @ KX.unsqueeze(-1)).squeeze(-1)

    return X_hat.to('cpu'), {"KX": KX.to('cpu'), "AeKX": AeKX.to('cpu'), "history": history}



# %% id="-5ZGafdi_WX2"
fres = np.load("/home/yousen/Public/ndlar_shared/data/response_v2a_distance_10p431cm_binsize_0p04434cm_tick0p05us.npy")

# %% colab={"base_uri": "https://localhost:8080/"} id="A-qtwOyfF9G_" outputId="4b922345-3b79-465a-a646-e78a949dcfa2"
# rebin to 100 ns
freslen = 225 + 1
fres_start = 450 * 2 - 1 * 2
fres_end = fres_start + freslen * 2

fres3x3 = np.empty((3,3, freslen)) # rebin to 100ns
# center at [1,1]
# [[0,0],[0,1], [0,2],
#  [1,0], [1,1], [1,2],
#  [2,0], [2,1], [2,2]]

resplane_shift = (fres_start + 2) // 2
fres3x3[1,1] = np.mean(fres[:5,:5], axis=(0,1))[fres_start:fres_end:2]
fres3x3[2,1] = np.mean(fres[:5,5:10], axis=(0,1))[fres_start:fres_end:2]
fres3x3[1,2] = np.mean(fres[5:10,:5], axis=(0,1))[fres_start:fres_end:2]
fres3x3[2,2] = np.mean(fres[5:10,5:10], axis=(0,1))[fres_start:fres_end:2]
fres3x3[0,0] = fres3x3[2,2]
fres3x3[0,1] = fres3x3[2,1]
fres3x3[1,0] = fres3x3[1,2]
fres3x3[0,2] = fres3x3[2,2]
fres3x3[2,0] = fres3x3[2,2]
fres3x3 = fres3x3/np.sum(fres3x3[1,1])
print(fres3x3.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 448} id="1S3NCpyNGL8_" outputId="cddc608c-05a1-48b3-8a77-162942dbed56"
for i in range(3):
    for j in range(3):
        plt.plot(fres3x3[i,j], label=f"[{i},{j}]")
plt.legend()

# %% id="hY_KQY9aJZrO"
finput = np.load("tests/playground/for_sp/single_track_full_fr_noises_20250924.npz")

# %% colab={"base_uri": "https://localhost:8080/"} id="_rBNyfPQK1aq" outputId="9826e99d-2fb6-4b48-9183-42c27f9b3ebe"
# finput.files

# %% colab={"base_uri": "https://localhost:8080/"} id="1Lf3HY-HLAOG" outputId="ccc46a99-c675-42a8-9ae8-e4c4eeadfead"
finput['hits_tpc2_batch11'], finput['one_tick'], finput['time_spacing']

# %% id="x8--KT_zLDix"
effql = torch.tensor(finput['effq_tpc2_batch11_location'])
effq = torch.tensor(finput['effq_tpc2_batch11'][:,3])
effqb = Block(location=effql, data=effq.unsqueeze(1).unsqueeze(1).unsqueeze(1))


# %% id="7PEeKe70LdCD"
def coarse_chunk(block, shp_tensor):
    block =  accumulate(chunkify(block, shp_tensor))
    block = Block(data=block.data, location=block.location)
    return block

def block_on_sgrid(block, shp_tensor):
    block = coarse_chunk(block, shp_tensor)
    location = SGrid(shp_tensor).spoint(block.location)
    return Block(location=location, data=block.data.sum(dim=(1,2,3), keepdim=True))


# %% colab={"base_uri": "https://localhost:8080/"} id="nTYIS25nVnBd" outputId="51a5801c-b284-4883-ee98-2312045b8956"
# everything on 100ns basis

effqb_sgrid = block_on_sgrid(effqb, (1,1,2))
coarse_effqb_sgrid = coarse_chunk(effqb_sgrid, (1,1,25))
coarse_effqb_sgrid.location += torch.tensor([0,0,resplane_shift])
torch.min(effqb.location[:,-1]), torch.max(effqb.location[:,-1]), torch.min(coarse_effqb_sgrid.location[:,-1]), torch.max(coarse_effqb_sgrid.location[:,-1])

# %% id="FSNvadYxWVK4"

# %% colab={"base_uri": "https://localhost:8080/", "height": 542} id="eaRQMHJ4YjoS" outputId="d08f7e69-e3bb-4762-cc7c-d11f9c8a02fe"
# fig = go.Figure(data=go.Scatter3d(x=effqb_sgrid.location[:,0],
#                                   y=effqb_sgrid.location[:,1],
#                                   z=effqb_sgrid.location[:,2],
#                                   mode="markers", marker=dict(size=1, color=effqb_sgrid.data.squeeze())))
# fig.show()

# %% id="HQ9Gs0tQZaoT"
# convert hit from 50ns basis to 100ns basis
hits_loc = finput['hits_tpc2_batch11_location'][:,[0,1,3]]
hits_q = finput['hits_tpc2_batch11'][:,3]

# %% colab={"base_uri": "https://localhost:8080/"} id="VcvGumNjabeY" outputId="d761931b-053c-4772-f14e-fa7193cb2511"
finput['hits_tpc2_batch11_location'].shape

# %% id="v6K22-qEadny"
hb = Block(location=torch.tensor(hits_loc), data=torch.tensor(hits_q).unsqueeze(1).unsqueeze(1).unsqueeze(1))
hb_sgrid = block_on_sgrid(hb, (1,1,2))

# %% colab={"base_uri": "https://localhost:8080/", "height": 542} id="5AufppYSavH9" outputId="2914d843-f855-437f-e735-baaa9ec9f17e"
# fig = go.Figure(data=go.Scatter3d(x=hb_sgrid.location[:,0],
#                                   y=hb_sgrid.location[:,1],
#                                   z=hb_sgrid.location[:,2],
#                                   mode="markers", marker=dict(size=1, color=hb_sgrid.data.squeeze())))
# fig.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="ORc_CR8vcBay" outputId="e6fa2245-7e30-451f-cb4e-181029a4e381"
# generate masks

tshift = np.argmax(fres3x3[1,1])
support_trange = 25
pre_n = 15
post_n = 5
thres = 5
adc_hold_delay_ticks = 15
csa_reset_ticks = 1

def support_matrices(support_trange=support_trange, pre_n=pre_n, post_n=post_n, tshift=tshift,
                     hitblock = hb_sgrid, thres = thres, adc_hold_delay_ticks = adc_hold_delay_ticks, csa_reset_ticks = csa_reset_ticks,
                     freslen=401):
    tmin = torch.min(hitblock.location[:,-1] - tshift - pre_n) // support_trange * support_trange
    tmax = (torch.max(hitblock.location[:,-1] - tshift + post_n) // support_trange + 1)* support_trange


    xmin = torch.min(hitblock.location[:,0])
    xmax = torch.max(hitblock.location[:,0])+1
    ymin = torch.min(hitblock.location[:,1])
    ymax = torch.max(hitblock.location[:,1])+1

    hitblock_loc_on_min = Block(location=hitblock.location - torch.tensor([xmin, ymin, tmin]), data=hitblock.data)
    hitblock_loc_on_min_tshift = Block(location=hitblock.location - torch.tensor([xmin, ymin, tmin+tshift]), data=hitblock.data)

    mhd = torch.zeros((xmax-xmin, ymax-ymin, tmax-tmin + (tshift // support_trange + 1) * support_trange), dtype=torch.bool)
    hqblock = accumulate(chunkify(hitblock_loc_on_min, mhd.shape))
    mqblock = accumulate(chunkify(hitblock_loc_on_min_tshift, (xmax-xmin, ymax-ymin, tmax-tmin)))
    mhblock = Block(location=hqblock.location, data=hqblock.data>0)
    qinitial = mqblock.data.clone().detach()
    mqblock.data = mqblock.data > 0
    mqexpandblock = Block(data=mqblock.data.clone().detach(), location=mqblock.location.clone().detach())

    for i in range(mqblock.shape[0]):
        for j in range(mqblock.shape[1]):
            for k in range(mqblock.shape[2]):
                if mqblock.data[0,i,j,k]:
                    mqexpandblock.data[0,i-1:i+1, j-1:j+1, k-pre_n:k+post_n] = True

    nhits = torch.max(torch.sum(mhblock.data, dim=-1)) + 1

    Ae = torch.zeros((1, mqblock.shape[0], mqblock.shape[1], nhits, mqblock.shape[2]+freslen-1))
    A0 = torch.tril(torch.ones((mqblock.shape[2]+freslen-1, mqblock.shape[2]+freslen-1)), diagonal=1).expand(
        1, mqblock.shape[0], mqblock.shape[1], mqblock.shape[2]+freslen-1, mqblock.shape[2]+freslen-1
    ).clone().detach()

    hq = torch.zeros((1, mqblock.shape[0], mqblock.shape[1], nhits))
    last_timestamp = torch.full((1, mqblock.shape[0], mqblock.shape[1], 1), fill_value=1E-9, dtype=torch.int)

    start_timeindex = torch.argmax(mhblock.data.to(torch.int), dim=-1, keepdim=True)
    end_timeindex = mhblock.shape[-1] - 1 - torch.argmax(mhblock.data.to(torch.int).flip(dims=(-1,)), dim=-1, keepdim=True)
        # Use broadcasting to set elements in A0
    trange = torch.arange(A0.shape[-1], device=A0.device)
    start_timeindex_broadcast = start_timeindex.unsqueeze(-1).expand_as(A0)
    end_timeindex_broadcast = end_timeindex.unsqueeze(-1).expand_as(A0)
    trange_broadcast = trange.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(A0)

    mask = (trange_broadcast >= start_timeindex_broadcast) & (trange_broadcast <= end_timeindex_broadcast)
    A0 = torch.where(mask, torch.zeros_like(A0), A0)

    for i in range(nhits):
        trange = torch.arange(mqblock.shape[2]+fres3x3.shape[-1]-1)
        timestamp_idx = torch.argmax(mhblock.data.to(torch.int), dim=-1, keepdim=True)
        src = torch.zeros_like(timestamp_idx, dtype=mhblock.data.dtype, device=mhblock.device)
        ishit = torch.gather(mhblock.data, dim=-1, index=timestamp_idx)
        timestamp_idx[~ishit] = -1E9
        if i == 0:
            Ae[..., i, :] = trange[None, None, None, None, :] + tmin <= (timestamp_idx - adc_hold_delay_ticks)
            hq[..., i] = thres
        if i == 1:
            Ae[..., i, :] = (
                (trange[None, None, None, None, :] <= (timestamp_idx))
                & (trange[None, None, None, None, :] > (timestamp_idx - adc_hold_delay_ticks))
            )
            hq[..., i] =  torch.where(
                ishit.squeeze(-1),
                torch.gather(hqblock.data, dim=-1, index=
                         torch.clamp(timestamp_idx, min=0)).squeeze(-1) - thres,
                0,
            )
            last_timestamp = timestamp_idx.clone().detach()
            mhblock.data.scatter_(dim=-1, index=torch.where(ishit, timestamp_idx, 0), src=src)
        if i > 1:
            Ae[..., i, :] = (trange[None, None, None, None, :] <=  timestamp_idx) & (
                        trange[None, None, None, None, :] >= (last_timestamp + csa_reset_ticks)
            )
            hq[..., i] =  torch.where(
                ishit.squeeze(-1),
                torch.gather(hqblock.data, dim=-1, index=
                         torch.clamp(timestamp_idx, min=0)).squeeze(-1),
            0,
            )
            last_timestamp = timestamp_idx.clone().detach()
            mhblock.data.scatter_(dim=-1, index=torch.where(ishit, timestamp_idx, 0), src=src)

    return torch.tensor([xmin, ymin, tmin]), torch.tensor([xmax, ymax, tmax]), tshift, hqblock, mqblock, qinitial, mqexpandblock, mhblock, hq, Ae, A0

xyzmin, xyzmax, tshift, hqblock, mqblock, qinitial, mqexpandblock, mhblock, hq, Ae, A0 = support_matrices(freslen=freslen)

# %% id="5OcpkYDv4ydy"
X_hat, iterations =  solve_nonnegative_3d(Ae, A0, torch.tensor(fres3x3), hq, mqexpandblock.data, lam_l1=0.01, lam_dx=0.01, lam_a0=0.01,
                                          Z0=qinitial, smooth_kernel=gaussian_kernel1d(sigma=2.0, kernel_size=15, device='cuda'), device='cuda')

# %% id="MadXbnIY82db"
Xhatblock = Block(data=X_hat.to('cpu'), location=mqblock.location)
Xhatcoarseblock = coarse_chunk(Xhatblock, (1,1,25))
Xhatcoarseblock.location += xyzmin

# %% id="HRONd_cL9IpR"
# fig = go.Figure(data=go.Scatter3d(x=Xhatcoarseblock.location[:,0],
#                                   y=Xhatcoarseblock.location[:,1],
#                                   z=Xhatcoarseblock.location[:,2],
#                                   mode="markers", marker=dict(size=1, color=Xhatcoarseblock.data.sum(dim=(1,2,3)))))

# fig.add_trace(go.Scatter3d(x=coarse_effqb_sgrid.location[:,0],
#                                   y=coarse_effqb_sgrid.location[:,1],
#                                   z=coarse_effqb_sgrid.location[:,2],
#                                   mode="markers", marker=dict(size=1, color=coarse_effqb_sgrid.data.sum(dim=(1,2,3)))))
# fig.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 542} id="tHPJEOxA9Z28" outputId="635c9814-526a-44e9-c41f-70e3ceb7a37a"
fig = go.Figure(data=go.Scatter3d(x=Xhatcoarseblock.location[:,0],
                                  y=Xhatcoarseblock.location[:,1],
                                  z=Xhatcoarseblock.location[:,2],
                                  mode="markers", marker=dict(size=1)))

fig.add_trace(go.Scatter3d(x=coarse_effqb_sgrid.location[:,0],
                                  y=coarse_effqb_sgrid.location[:,1],
                                  z=coarse_effqb_sgrid.location[:,2],
                                  mode="markers", marker=dict(size=1)))

# # fig.add_trace(go.Scatter3d(x=hb_sgrid_loc_on_min_tshift.location[:,0],
# #                                   y=hb_sgrid_loc_on_min_tshift.location[:,1],
# #                                   z=hb_sgrid_loc_on_min_tshift.location[:,2],
# #                                   mode="markers", marker=dict(size=1)))
# fig.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 527} id="VssnicEefg8O" outputId="d4cd04f4-eb83-42cd-887c-74c0f3654002"
# Get all unique locations from both blocks
all_locations = torch.cat((Xhatcoarseblock.location, coarse_effqb_sgrid.location), dim=0)
unique_locations, _ = torch.unique(all_locations, dim=0, return_inverse=True)

# Create tensors to hold aligned data, initialized with zeros
Xhat_data_padded = torch.zeros((unique_locations.shape[0],) + Xhatcoarseblock.data.shape[1:], dtype=Xhatcoarseblock.data.dtype)
effq_data_padded = torch.zeros((unique_locations.shape[0],) + coarse_effqb_sgrid.data.shape[1:], dtype=coarse_effqb_sgrid.data.dtype)

# Fill in data for existing locations
for i, loc in enumerate(unique_locations):
    # Find index in Xhatcoarseblock
    Xhat_match_indices = torch.where(torch.all(Xhatcoarseblock.location == loc, dim=1))[0]
    if len(Xhat_match_indices) > 0:
        Xhat_data_padded[i] = Xhatcoarseblock.data[Xhat_match_indices].squeeze(0)

    # Find index in coarse_effqb_sgrid
    effq_match_indices = torch.where(torch.all(coarse_effqb_sgrid.location == loc, dim=1))[0]
    if len(effq_match_indices) > 0:
        effq_data_padded[i] = coarse_effqb_sgrid.data[effq_match_indices].squeeze(0)

print("Number of unique locations:", unique_locations.shape[0])
print("Shape of padded Xhat data:", Xhat_data_padded.shape)
print("Shape of padded effq data:", effq_data_padded.shape)
# Calculate the difference in summed data at each location
difference = Xhat_data_padded.sum(dim=(1,2,3)) - effq_data_padded.sum(dim=(1,2,3))

# Plot a histogram of the differences
plt.figure()
plt.hist(difference.numpy(), bins=50)
plt.xlabel("Difference in Summed Data")
plt.ylabel("Frequency")
plt.title("Histogram of Differences between Xhatcoarseblock and coarse_effqb_sgrid (Aligned)")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="5258b2f2" outputId="7a7863b1-9a5e-4d1e-ce27-a57ba1136669"
# Get all unique 2D locations from all three blocks
all_locations_2d = torch.cat((Xhatcoarseblock.location[:, :2], coarse_effqb_sgrid.location[:, :2], hb_sgrid.location[:,:2]), dim=0)
unique_locations_2d, _ = torch.unique(all_locations_2d, dim=0, return_inverse=True)

# Create dictionaries to hold summed data for each unique 2D location, initialized with zeros
xhat_grouped_sum_2d = {tuple(loc.tolist()): 0.0 for loc in unique_locations_2d}
effq_grouped_sum_2d = {tuple(loc.tolist()): 0.0 for loc in unique_locations_2d}
hb_grouped_sum_2d = {tuple(loc.tolist()): 0.0 for loc in unique_locations_2d}


# Populate dictionaries with summed data from Xhatcoarseblock
for i in range(Xhatcoarseblock.location.shape[0]):
    location_key_2d = tuple(Xhatcoarseblock.location[i, :2].tolist())
    xhat_grouped_sum_2d[location_key_2d] += Xhatcoarseblock.data[i].sum().item()

# Populate dictionaries with summed data from coarse_effqb_sgrid
for i in range(coarse_effqb_sgrid.location.shape[0]):
    location_key_2d = tuple(coarse_effqb_sgrid.location[i, :2].tolist())
    effq_grouped_sum_2d[location_key_2d] += coarse_effqb_sgrid.data[i].sum().item()

# Populate dictionaries with summed data from hb_sgrid
for i in range(hb_sgrid.location.shape[0]):
    location_key_2d = tuple(hb_sgrid.location[i, :2].tolist())
    hb_grouped_sum_2d[location_key_2d] += hb_sgrid.data[i].sum().item()


# Convert dictionaries to lists of summed data, maintaining the order of unique_locations_2d
xhat_summed_data_aligned = torch.tensor([xhat_grouped_sum_2d[tuple(loc.tolist())] for loc in unique_locations_2d])
effq_summed_data_aligned = torch.tensor([effq_grouped_sum_2d[tuple(loc.tolist())] for loc in unique_locations_2d])
hb_summed_data_aligned = torch.tensor([hb_grouped_sum_2d[tuple(loc.tolist())] for loc in unique_locations_2d])


# Calculate the difference
difference_2d_sum = xhat_summed_data_aligned - effq_summed_data_aligned
difference_2d_hits_sum = hb_summed_data_aligned - effq_summed_data_aligned

over_threshold = effq_summed_data_aligned > thres
difference_2d_sum_over_threshold = difference_2d_sum[over_threshold]
difference_2d_hits_sum_over_threshold = difference_2d_hits_sum[over_threshold]


# Plot the histogram of differences
plt.figure()
plt.hist(difference_2d_sum.numpy(), bins=50, alpha=0.5, label='sp')
plt.hist(difference_2d_hits_sum.numpy(), bins=50, alpha=0.5, label='raw')
plt.legend()
plt.xlabel("Difference in Summed Data (Grouped by Pixel ID)")
plt.ylabel("Frequency")
plt.title("Histogram of Differences (Grouped by Pixel ID)")
plt.show()

plt.figure()
plt.hist(difference_2d_sum_over_threshold.numpy(), bins=50, alpha=0.5, label='sp')
plt.hist(difference_2d_hits_sum_over_threshold.numpy(), bins=50, alpha=0.5, label='raw')
plt.xlabel("Difference in Summed Data, Effq>thres (Grouped by Pixel)")
plt.ylabel("Frequency")
plt.title("Histogram of Differences, Effq>thres (Grouped by Pixel)")
plt.legend()
plt.show()

plt.figure()
plt.plot(effq_summed_data_aligned, xhat_summed_data_aligned, 'o', label='Xhat vs Effq')
plt.plot(effq_summed_data_aligned, hb_summed_data_aligned, 'x', label='Hb vs Effq')
plt.plot(np.arange(0, 30, 0.1), np.arange(0, 30, 0.1), '-')
plt.xlabel("Summed Q from Effq per pixel")
plt.ylabel("Summed Q per pixel")
plt.title("Scatter Plot of Summed Data from Xhatcoarseblock, Hb and coarse_effqb_sgrid")
plt.legend()
plt.show()

# %%
qfinegrain = accumulate(chunkify(Xhatblock, (1,1,Xhatblock.shape[-1])))

sumq = qfinegrain.data.sum(dim=-1)
qfinegrain = qfinegrain.data.view(-1, qfinegrain.shape[-1])
for i in range(qfinegrain.shape[0]):
    if i % 10 == 0:
        plt.figure()
    plt.plot(qfinegrain[i])

# %%
iterations['history']
steps = []
dloss = []
l1loss = []
dxloss = []
a0loss = []

for it in iterations['history']:
    steps.append(it['step'])
    dloss.append(it['data'])
    l1loss.append(it['l1'])
    dxloss.append(it['dx'])
    a0loss.append(it['a0'])
plt.plot(steps, dloss, label='data')
plt.plot(steps, l1loss, label='l1')
plt.plot(steps, dxloss, label='dx')
plt.plot(steps, a0loss, label='a0')
plt.legend()
plt.yscale('log')

# %%
