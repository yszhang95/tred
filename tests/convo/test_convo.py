import torch
from tred.response import ndlarsim, quadrant_copy
from tred.convo import symmetric_pad, convolve, interlaced
# from tred.convo import interlaced_symm_v2 as interlaced_symm
from tred.convo import interlaced_symm_v2, interlaced_symm
from tred.blocking import Block
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from torch.nn.functional import conv2d, conv3d, pad

def plot_symmetric_pad_1d():
    # (odd, even) and (odd, even)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8*2, 8*2))
    for i, nt in enumerate(range(4, 6)):
        for j, nz in enumerate(range(3,5)):
            t = torch.arange(nt) + 1
            shape = (nt+nz,)
            for k, style in enumerate(['append', 'prepend', 'center', 'edge']):
                tp = symmetric_pad(t, shape, (style,))
                axes[i,j].stem(np.arange(shape[0])+0.05*k, tp.numpy(), linefmt=f'C{k}-', basefmt=" ", label=style)
            axes[i,j].legend()
            axes[i,j].axhline(0, color='gray', linestyle='--', linewidth=1)
            axes[i,j].set_title(f'nt={nt}, nz={nz}')

    fig.savefig('symmetric_pad_1d.png')

def plot_sr_pad_1d():
    # (odd, even) and (odd, even)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8*2, 8*2))
    for i, ns in enumerate(range(4, 6)):
        for j, nr in enumerate(range(3,5)):
            s = 2*torch.ones((ns,))
            r = 4*torch.ones((nr,))
            shape = (ns+nr-1,)
            sp = symmetric_pad(s, shape, ('edge',))
            rp = symmetric_pad(r, shape, ('center',))
            axes[i,j].stem(np.arange(shape[0])-0.05, sp.numpy(), linefmt=f'C{0}-', basefmt=" ", label='edge')
            axes[i,j].stem(np.arange(shape[0])+0.05, rp.numpy(), linefmt=f'C{4}-', basefmt=" ", label='center')
            axes[i,j].legend()
            axes[i,j].axhline(0, color='gray', linestyle='--', linewidth=1)
            axes[i,j].set_title(f'ns={ns}, nr={nr}, shape={shape[0]}')

    fig.savefig('sr_pad_1d.png')

def direct_convo_pixel_time_2d(data, kernel, flip=True):
    '''
    Input are unbatched
    '''
    dims = torch.arange(kernel.ndim).tolist()
    kernel_new = kernel.flip(dims=dims) if flip else kernel.clone().detach()
    nz = [[i-1, i-1] for i in kernel.size()]
    nz = [j for i in nz for j in i]
    nz = nz[::-1] # last axis appears in the front
    data_new = pad(data, nz, 'constant', 0) # 2D
    data_new = data_new.unsqueeze(0).unsqueeze(1)
    kernel_new = kernel_new.unsqueeze(0).unsqueeze(0)
    return conv2d(data_new, kernel_new, padding='valid').squeeze(1)

def direct_hybrid_conv_corr_nd_unbatch(data, kernel, taxis, nz_data, stride=(10,10,1)):
    '''
    data is a 3D-tensor, (10*nx, 10*ny, nt)
    kernel is a 3D-tensor, (10*nx, 10*ny, nt)
    conv along taxis, cross correlation for others
    '''
    # channels 1, batch 1
    data_new = pad(data, nz_data, 'constant', 0) # 3D
    data_new = data_new.unsqueeze(0).unsqueeze(1)
    # channels 1, batch 1
    kernel = kernel.flip(dims=(taxis,)).unsqueeze(0).unsqueeze(1).contiguous()
    # channels, 1, strides (10, 10, 1)
    return conv3d(data_new, kernel, padding='valid', stride=stride).squeeze(1)

def test_nd():
    nimpx, nimpy = 10, 10

    data = torch.arange(2*nimpx*2*nimpy*2).view(2*nimpx, 2*nimpy, 2).to(torch.float32)

    ndpath = "response_38_v2b_50ns_ndlar.npy"
    kernel_conv = ndlarsim(ndpath)

    quadrant = torch.tensor(np.load(ndpath).astype(np.float32))
    kernel = quadrant_copy(quadrant)

    kernel_conv = kernel_conv[:,:,6000:6100]
    kernel_conv = kernel_conv.contiguous()
    kernel = kernel[:,:,6000:6100]
    kernel = kernel.contiguous()

    nx, ny, nt = kernel.shape
    nz = (nt-1, nt-1, (ny//nimpy-1)*nimpy, (ny//nimpy-1)*nimpy,
          (nx//nimpx-1)*nimpx, (nx//nimpx-1)*nimpx,)
    results = direct_hybrid_conv_corr_nd_unbatch(data=data, kernel=kernel,
                                           taxis=-1, nz_data=nz,
                                           stride=(nimpx,nimpy,1))
    signal = Block(location=torch.tensor([0,0,0]), data=data)
    output = interlaced(signal, kernel_conv, steps=torch.tensor((nimpx,nimpy,1)), taxis=-1)
    assert torch.allclose(output.data, results, atol=1E-5, rtol=1E-5)


def test_convo():

    q1 = torch.arange(2) + 5
    q2 = torch.arange(3) + 5
    q3 = torch.arange(4) + 5

    q1 = q1.repeat(2).view(2, -1).T
    q2 = q2.repeat(2).view(2, -1).T
    q3 = q3.repeat(2).view(2, -1).T

    location = torch.tensor([[0,0]]*len(q1))

    s1 = Block(location=torch.tensor([[0,0]]), data=q1)
    s2 = Block(location=torch.tensor([[0,0]]), data=q2)
    s3 = Block(location=torch.tensor([[0,0]]), data=q3)

    response = torch.tensor([[1, 1], [2, 2], [1, 1]]) # center in the center; length along pixel domain is always odd
    response_conv = response

    for q, s in zip([q1, q2, q3], [s1, s2, s3]):
        result = convolve(s, response_conv)
        output = direct_convo_pixel_time_2d(q, response)

        assert torch.allclose(result.data, output.to(torch.float32))

def plot_convolve_2d():
    # Needs to include an extra dimension due to annoying setup of padding in convo.py. .
    q = torch.tensor([[1,0], [0,1]]) # non-batched, unit charge;
    response = torch.tensor([[1,0], [2,0], [1,0]]) # 3 pixels; nothing in the next moment

    response_conv = response

    signal = Block(location=torch.tensor([[0,0]]), data=q)

    result = convolve(signal, response_conv)
    X = result.data[0].numpy().T
    loc = result.location[0][[1,0]]

    x = np.arange(loc[1], loc[1]+X.shape[1], 1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8*2, 8*2))
    axes = axes.flat
    for i in range(X.shape[0]):
        axes[i].bar(x, height=X[i])
        axes[i].set_title(f'time == {loc[0]+i}')
        # Force integer ticks on y-axis
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Force integer ticks on x-axis (if needed)
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add the text inside the plot, anchored to the top-left (axes coordinates: (0, 1))
    axes[3].text(
        0.2, 0.8,  # x, y position in axes fraction (0 = left/bottom, 1 = right/top)
        f"Initial location on lower-left corner {signal.location[0].tolist()}\n"
        f"Initial Q {signal.data[0].tolist()}\n"
        f"taxis=-1\n"
        f"Response from collection pixel\n {response_conv.tolist()}\n"
        f"Symmetric response \n{response.tolist()}\n",
        fontsize=12,
        va='top',   # vertical alignment
        ha='left',  # horizontal alignment
        transform=axes[3].transAxes,  # so (0,1) refers to axes coordinates, not data
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')  # optional, to make it more readable
    )

    fig.savefig('unitq_convolve_2d.png')


    # Needs to include an extra dimension due to annoying setup of padding in convo.py. .
    q = torch.tensor([[1,0], [0,0], [0,0], [1,0], [0,1]]) # non-batched, unit charge;
    response = torch.tensor([[1,0], [2,0], [1,0]]) # 3 pixels; nothing in the next moment

    response_conv = response

    signal = Block(location=torch.tensor([[0,0]]), data=q)

    result = convolve(signal, response_conv)
    X = result.data[0].numpy().T
    loc = result.location[0][[1,0]]

    x = np.arange(loc[1], loc[1]+X.shape[1], 1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8*2, 8*2))
    axes = axes.flat
    for i in range(X.shape[0]):
        axes[i].bar(x, height=X[i])
        axes[i].set_title(f'time == {loc[0]+i}')
        # Force integer ticks on y-axis
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Force integer ticks on x-axis (if needed)
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add the text inside the plot, anchored to the top-left (axes coordinates: (0, 1))
    axes[3].text(
        0.2, 0.8,  # x, y position in axes fraction (0 = left/bottom, 1 = right/top)
        f"Initial location on lower-left corner {signal.location[0].tolist()}\n"
        f"Initial Q {signal.data[0].tolist()}\n"
        f"taxis=-1\n"
        f"Response from collection pixel\n {response_conv.tolist()}\n"
        f"Symmetric response \n{response.tolist()}\n",

        fontsize=12,
        va='top',   # vertical alignment
        ha='left',  # horizontal alignment
        transform=axes[3].transAxes,  # so (0,1) refers to axes coordinates, not data
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')  # optional, to make it more readable
    )

    fig.savefig('unitq_convolve_2d_long.png')

def test_interlaced():

    q1 = torch.arange(2) + 5
    q2 = torch.arange(3) + 5
    q3 = torch.arange(4) + 5

    q1 = q1.repeat(2).repeat_interleave(2).reshape(2,-1,2).permute(1,0,2).reshape(-1,2)
    q2 = q2.repeat(2).repeat_interleave(2).reshape(2,-1,2).permute(1,0,2).reshape(-1,2)
    q3 = q3.repeat(2).repeat_interleave(2).reshape(2,-1,2).permute(1,0,2).reshape(-1,2)

    s1 = Block(location=torch.tensor([[0,0]]), data=q1)
    s2 = Block(location=torch.tensor([[0,0]]), data=q2)
    s3 = Block(location=torch.tensor([[0,0]]), data=q3)

    response = torch.tensor([[1, 1], [1,1], [2, 2],[2,2], [1, 1], [1,1]]) # center in the center; length along pixel domain is always odd
    response_conv = response

    for q, s in zip([q1, q2, q3], [s1, s2, s3]):
        result = interlaced(s, response_conv, steps=torch.tensor([2,1], dtype=torch.int32), taxis=-1)
        output = direct_convo_pixel_time_2d(q.view(-1,2,2)[:,0,:], response.view(-1,2,2)[:,0,:])

        assert torch.allclose(result.data, 2*output.to(torch.float32))

        assert torch.equal(result.location[0], torch.tensor([-1,0]))

def test_nd_symm():

    nimpx, nimpy = 10, 10

    data = torch.arange(2*nimpx*2*nimpy*2).view(2*nimpx, 2*nimpy, 2).to(torch.float32)

    ndpath = "response_38_v2b_50ns_ndlar.npy"
    kernel_conv = ndlarsim(ndpath)

    quadrant = torch.tensor(np.load(ndpath).astype(np.float32))
    kernel = quadrant_copy(quadrant)

    kernel_conv = kernel_conv[:,:,6000:6100]
    kernel_conv = kernel_conv.contiguous()
    kernel = kernel[:,:,6000:6100]
    kernel = kernel.contiguous()

    nx, ny, nt = kernel.shape
    nz = (nt-1, nt-1, (ny//nimpy-1)*nimpy, (ny//nimpy-1)*nimpy,
          (nx//nimpx-1)*nimpx, (nx//nimpx-1)*nimpx,)
    results = direct_hybrid_conv_corr_nd_unbatch(data=data, kernel=kernel,
                                           taxis=-1, nz_data=nz,
                                           stride=(nimpx,nimpy,1))
    data = data.repeat((2,1,1,1))
    data[1] = 2*data[1]
    signal = Block(location=torch.tensor([[0,0,0],[0,0,0]]), data=data)

    output = interlaced_symm(signal, kernel_conv, steps=torch.tensor((nimpx,nimpy,1)), taxis=-1, symm_axis=0)
    for i in range(2):
        assert torch.allclose(output.data[i], (i+1)*results, atol=1E-5, rtol=1E-5)
    output = interlaced_symm(signal, kernel_conv, steps=torch.tensor((nimpx,nimpy,1)), taxis=-1, symm_axis=1)
    for i in range(2):
        assert torch.allclose(output.data[i], (i+1)*results, atol=1E-5, rtol=1E-5)
    output = interlaced_symm_v2(signal, kernel_conv, steps=torch.tensor((nimpx,nimpy,1)), taxis=-1, symm_axis=0)
    for i in range(2):
        assert torch.allclose(output.data[i], (i+1)*results, atol=1E-5, rtol=1E-5)
    output = interlaced_symm_v2(signal, kernel_conv, steps=torch.tensor((nimpx,nimpy,1)), taxis=-1, symm_axis=1)
    for i in range(2):
        assert torch.allclose(output.data[i], (i+1)*results, atol=1E-5, rtol=1E-5)

    output2 = interlaced(signal, kernel_conv, steps=torch.tensor((nimpx,nimpy,1)), taxis=-1)
    assert torch.allclose(output2.data[0], results, atol=1E-5, rtol=1E-5)
    assert torch.equal(output.location, output2.location)


def plot_interlaced_2d():
    # Needs to include an extra dimension due to annoying setup of padding in convo.py. .
    q = torch.tensor([[1,0], [2,0], [0,0], [0,0], [0,0], [0,0], [1,0], [1,0], [0,1], [0,1]]) # non-batched, unit charge;
    response = torch.tensor([[0.5,0], [1,0], [2,0], [2,0], [1,0], [0.5,0]]) # 3 pixels; nothing in the next moment

    response_conv = response

    signal = Block(location=torch.tensor([[0,0]]), data=q.to(torch.float32))

    result = interlaced_symm_v2(signal, response_conv, steps=torch.tensor([2,1], dtype=torch.int32), taxis=-1)

    X = result.data[0].numpy().T
    loc = result.location[0][[1,0]]

    x = np.arange(loc[1], loc[1]+X.shape[1], 1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8*2, 8*2))
    axes = axes.flat
    for i in range(X.shape[0]):
        axes[i].bar(x, height=X[i])
        axes[i].set_title(f'time == {loc[0]+i}')
        # Force integer ticks on y-axis
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Force integer ticks on x-axis (if needed)
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add the text inside the plot, anchored to the top-left (axes coordinates: (0, 1))
    axes[3].text(
        0.01, 0.8,  # x, y position in axes fraction (0 = left/bottom, 1 = right/top)
        f"Initial location on lower-left corner {signal.location[0].tolist()}\n"
        f"Initial Q {signal.data[0].tolist()}\n"
        f"taxis=-1\n"
        f"Response from collection pixel\n {response_conv.tolist()}\n"
        f"Symmetric response \n{response.tolist()}\n",
        fontsize=12,
        va='top',   # vertical alignment
        ha='left',  # horizontal alignment
        transform=axes[3].transAxes,  # so (0,1) refers to axes coordinates, not data
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')  # optional, to make it more readable
    )

    fig.savefig('interlaced_convolve_2d.png')

def test_sym_convo1d():
    response = list(range(6))
    response += response[::-1]
    response = np.array(response)
    Q = np.arange(20)
    nr = response.shape[0]
    nq = Q.shape[0]
    # 4 partitions
    # 2 groups
    # group 1
    o1 = np.convolve(Q[3::4], response[3::4])
    o4 = np.convolve(Q[-1::-4], response[-1::-4])
    assert np.equal(o1, o4[::-1]).all()
    # group 2
    o2 = np.convolve(Q[2::4], response[2::4])
    o3 = np.convolve(Q[-2::-4], response[-2::-4])
    assert np.equal(o2, o3[::-1]).all()


if __name__ == '__main__':
    plot_symmetric_pad_1d()
    plot_sr_pad_1d()
    plot_convolve_2d()
    plot_interlaced_2d()

    test_convo()
    test_nd()
    test_interlaced()
    test_sym_convo1d()
    test_nd_symm()
