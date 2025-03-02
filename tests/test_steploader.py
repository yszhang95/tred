from tred.loaders import StepLoader, _equal_div_script, steps_from_ndh5

import h5py
import matplotlib
import matplotlib.pyplot as plt

import torch
import numpy as np

import os
import sys
import time

def plot_segments(fpath):
    f = h5py.File(fpath, 'r')
    segments = f['segments']
    steploader = StepLoader(f, transform=steps_from_ndh5)
    # test for-loop
    x = {
        '_getitem__' : [],
        'get_column' : [],
        'h5py' : []
    }
    y = {
        '__getitem__' : [],
        'get_column' : [],
        'h5py' : []
    }
    z = {
        '__getitem__' : [],
        'get_column' : [],
        'h5py' : []
    }

    x['__getitem__'], y['__getitem__'], z['__getitem__']  = [
        torch.stack([steploader[:][0][:,2+i], steploader[:][0][:,5+i]], dim=1).view(-1) for i in range(3)
    ]
    x['get_column'] , y['get_column'], z['get_column']= [
        torch.stack([steploader.get_column(f'{c}_start'), steploader.get_column(f'{c}_end')], dim=1).view(-1) for c in ['x', 'y', 'z']
    ]
    x['h5py'] , y['h5py'], z['h5py']= [
        np.column_stack([segments[f'{c}_start'], segments[f'{c}_end']]).reshape(-1) for c in ['x', 'y', 'z']
    ]

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6*2, 6*2))

    axes[0,0].plot(x['__getitem__'], y['__getitem__'], 'o-', label='by __getitem__')
    axes[0,1].plot(z['__getitem__'], y['__getitem__'], 'o-', label='by __getitem__')
    axes[1,0].plot(x['__getitem__'], z['__getitem__'], 'o-', label='by __getitem__')

    axes[0,0].plot(x['get_column'], y['get_column'], '--', label='by get_column')
    axes[0,1].plot(z['get_column'], y['get_column'], '--', label='by get_column')
    axes[1,0].plot(x['get_column'], z['get_column'], '--', label='by get_column')

    axes[0,0].plot(x['h5py'], y['h5py'], '-.', label='by h5py')
    axes[0,1].plot(z['h5py'], y['h5py'], '-.', label='by h5py')
    axes[1,0].plot(x['h5py'], z['h5py'], '-.', label='by h5py')

    axes[0,0].legend()
    axes[0,1].legend()
    axes[1,0].legend()

    axes[0,0].set_xlabel('x[cm]')
    axes[0,0].set_ylabel('y[cm]')
    axes[0,1].set_xlabel('z[cm]')
    axes[0,1].set_ylabel('y[cm]')
    axes[1,0].set_xlabel('x[cm]')
    axes[1,0].set_ylabel('z[cm]')
    axes[1,1].text(0.1, 0.9, f'Input: {fpath}', wrap=True)

    fig.savefig('steploader_segments.png')

def test_special_numbers():
    data = [
        (1, 2, 1,1,1,2,4,9, 1, 1, 20),
        (1, 1, 2,4,9,2,4,10, 1, 2, 20)
    ]

    X0X1 = np.array(data)[:,2:8].astype(np.float32)
    X0X1 = torch.tensor(X0X1)
    dE = np.array(data)[:,0].astype(np.float32)
    dE = torch.tensor(dE)
    dEdx = np.array(data)[:,1].astype(np.float32)
    dEdx = torch.tensor(dEdx)
    pdg_id = np.array(data)[:,8].astype(np.int32)
    pdg_id = torch.tensor(pdg_id)
    event_id = np.array(data)[:,9].astype(np.int32)
    event_id = torch.tensor(event_id)
    t0_start = np.array(data)[:,10].astype(np.float64)
    t0_start = torch.tensor(t0_start)
    dummy = torch.empty((len(t0_start),0), dtype=torch.float64)

    dtype = np.dtype([('dE', 'f4'), ('dEdx', 'f4'), ('x_start', 'f4'),
                      ('y_start', 'f4'), ('z_start', 'f4'), ('x_end', 'f4'),
                      ('y_end', 'f4'), ('z_end', 'f4'),
                      ('pdg_id', 'i4'), ('event_id', 'i4'), ('t0_start', 'f8')])

    data = np.array(data, dtype=dtype)

    print('Initial data')
    print(data)

    step_limit = 1

    print('Test _equal_div')
    Ns = (torch.linalg.norm(X0X1[:,:3]-X0X1[:,3:], dim=1)//step_limit)\
        .to(torch.int32) + 1
    X0X1_, dE_, dEdx_, dummy_, ts_, intids_ = (
        _equal_div_script(X0X1, dE, dEdx, dummy, t0_start,
                          torch.stack([event_id, pdg_id], dim=1).view(-1, 2),
                          Ns)
    )

    print('Test _batch_equal_div')
    _X0X1, _dE, _dEdx, _dummy, _ts, _intids = (
        StepLoader._batch_equal_div(X0X1, dE, dEdx, dummy, t0_start,
                                    torch.stack([event_id, pdg_id], dim=1).view(-1, 2),
                                    step_limit, mem_limit=1/1024/1024,
                                    fn=_equal_div_script)
    )
    assert X0X1_.allclose(_X0X1)
    assert dE_.allclose(_dE)
    assert dEdx_.allclose(_dEdx)
    assert ts_.allclose(_ts)
    assert dummy_.allclose(_dummy)
    assert intids_.allclose(_intids)

    data = StepLoader(data, step_limit=step_limit, transform=steps_from_ndh5)
    print('New length', len(data))
    for i in range(len(data)):
        X0 = data[i][0][2:5]
        X1 = data[i][0][5:]
        dL =torch.linalg.norm(X0-X1)
        dE = data[i][0][0]
        dEdx = data[i][0][1]
        ts = data[i][1]
        intids = data[i][2]
        assert dL < step_limit
        assert X0.allclose(X0X1_[i,:3])
        assert X1.allclose(X0X1_[i,3:])
        assert dE.allclose(dE_[i])
        assert dEdx.allclose(dEdx_[i])
        assert ts.allclose(ts_[i])
        assert intids.equal(intids_[i])
        if i>0:
            assert data[i][0][2:5].allclose(data[i-1][0][5:8])

def test_nb():
    # try:
    #     raise ValueError
    # except Exception as e:
    try:
        import numba as nb
        print('\n----------------------------------------\nTesting numba')
        # numba based
        @nb.njit
        def chop_nb(X0X1, extf, intf, inti, Ns):
            # Example large N
            # Compute the total size of the output array
            total_size = np.sum(Ns)
            extf_dim = 1 # hard code
            intf_dim = 1 # hard code
            inti_dim = 2 # hard code

            # Preallocate a single output array
            x0x1_ = np.empty((total_size, 6), dtype=np.float32)
            extf_ = np.empty((total_size, ), dtype=np.float32)
            intf_ = np.empty((total_size, ), dtype=np.float32)
            inti_ = np.empty((total_size, inti_dim), dtype=np.int32)

            # Fill it using a loop but with direct indexing (avoiding repeated concatenation)
            idx = 0
            for i in range(len(X0X1)):
                size = Ns[i]
                for j in range(3):
                    temp = np.linspace(X0X1[i,j], X0X1[i,j+3], size+1)
                    x0x1_[idx:idx+size,j] = temp[:-1]
                    x0x1_[idx:idx+size,j+3] = temp[1:]
                    inti_[idx:idx+size, :] = inti[i] # hard code
                    intf_[idx:idx+size] = intf[i] # hard code
                    extf_[idx:idx+size] = extf[i]/size # hard code
                idx += size  # Move the index forward
            return x0x1_, extf_, intf_, inti_

        print('test nb')
        f = h5py.File('/home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5')
        segments = f['segments']
        data = {}
        data['float32'] = np.stack([segments[n]
                                    for n in ['dE', 'dEdx', 'x_start', 'y_start',
                                              'z_start', 'x_end', 'y_end', 'z_end']], axis=1).reshape(len(segments), -1)
        data['int32'] = np.stack([segments[n]
                                     for n in ['pdg_id', 'event_id']], axis=1)
        X0X1 = data['float32'][:,2:8]
        dE = data['float32'][:,0]
        dEdx = data['float32'][:,1]
        IntensiveInt = data['int32']
        step_limit = 1
        Ns = (np.linalg.norm(X0X1[:,:3]-X0X1[:,3:], axis=1) // step_limit).astype(np.int32) + 1
        chop_nb(X0X1, dE, dEdx, IntensiveInt, Ns)
        chop_nb(X0X1, dE, dEdx, IntensiveInt, Ns)
        chop_nb(X0X1, dE, dEdx, IntensiveInt, Ns)
        chop_nb(X0X1, dE, dEdx, IntensiveInt, Ns)
        dt = []
        for i in range(10):
            start = time.time()
            chop_nb(X0X1, dE, dEdx, IntensiveInt, Ns)
            end = time.time()
            dt.append(1000*(end-start))
        print('nb takes',np.array(dt).mean(), 'ms on average')
        print('dt', dt)
    except Exception as e:
        print('Faield running test_nb')
        print(e)

def test_perf():
    print('\n-------------------------------------')
    print('Testing performance of torch')
    f = h5py.File('/home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5')
    segments = f['segments']
    data = {}
    data['float32'] = torch.stack([torch.tensor(segments[n], requires_grad=False)
                                   for n in ['dE', 'dEdx', 'x_start', 'y_start',
                                             'z_start', 'x_end', 'y_end', 'z_end']], dim=1)
    data['float64'] = torch.tensor(segments['t0_start'])
    data['int32'] = torch.stack([torch.tensor(segments[n], dtype=torch.int32,
                                              requires_grad=False)
                                 for n in ['pdg_id', 'event_id']], dim=1)
    X0X1 = data['float32'][:,2:8]
    dE = data['float32'][:,0]
    dEdx = data['float32'][:,1]
    ExtensiveDouble = torch.empty((len(dE),0), dtype=torch.float64, requires_grad=False)
    IntensiveDouble = data['float64']
    IntensiveInt = data['int32']
    step_limit = 1
    mem_limit = 5*1024 # MB

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    StepLoader._batch_equal_div(X0X1, dE, dEdx, ExtensiveDouble, IntensiveDouble, IntensiveInt,
                                    step_limit, mem_limit, device='cuda',
                                fn=_equal_div_script)
    torch.cuda.synchronize()
    end.record()
    print(start.elapsed_time(end), 'ms on GPU, including data transfer from CPU to GPU')
    print('Peak cuda memory', torch.cuda.max_memory_allocated()/1024**2, 'MB')
    print('Peak cuda memory limit', mem_limit, 'MB')

    X0X1 = X0X1.to('cpu')
    dE = dE.to('cpu')
    dEdx = dEdx.to('cpu')
    IntensiveInt = IntensiveInt.to('cpu')
    mem_limit = 5*1024 # MB
    print('Warm up on CPU-GPU-CPU')
    for i in range(5):
        StepLoader._batch_equal_div(X0X1, dE, dEdx, ExtensiveDouble, IntensiveDouble, IntensiveInt,
                                    step_limit, mem_limit, device='cuda',
                                    fn=_equal_div_script)
    start = time.time()
    StepLoader._batch_equal_div(X0X1, dE, dEdx, ExtensiveDouble, IntensiveDouble, IntensiveInt,
                                step_limit, mem_limit, device='cuda',
                                fn=_equal_div_script)
    end = time.time()
    print('Elapsed', (end-start)*1E3, 'ms on CPU')

    X0X1cpu = X0X1.clone().detach()
    X0X1cpu.requires_grad = False
    X0X1 = X0X1.to('cuda:0')


    dE = dE.to('cuda:0')
    dEdx = dEdx.to('cuda:0')
    ExtensiveDouble = ExtensiveDouble.to('cuda:0')
    IntensiveDouble = IntensiveDouble.to('cuda:0')
    IntensiveInt = IntensiveInt.to('cuda:0')
    mem_limit = 5*1024 # MB
    print('Warm up on GPU')
    for i in range(5):
        StepLoader._batch_equal_div(X0X1, dE, dEdx, ExtensiveDouble, IntensiveDouble, IntensiveInt,
                                    step_limit, mem_limit, device='cuda',
                                    fn=_equal_div_script)
    torch.cuda.synchronize()
    start = time.time()
    X0X1new, _, _, _, _, _ = StepLoader._batch_equal_div(X0X1, dE, dEdx, ExtensiveDouble, IntensiveDouble, IntensiveInt,
                                                   step_limit, mem_limit, device='cuda',
                                                   fn = _equal_div_script)
    torch.cuda.synchronize()
    end = time.time()
    print('Elapsed', (end-start)*1000, 'ms on GPU, without data transfer')

    torch.cuda.synchronize()
    start = time.time()
    X0X1cpu.to('cuda').to('cpu')
    torch.cuda.synchronize()
    end = time.time()
    print('Elapsed', (end-start)*1000, 'ms for X0X1cpu transfering from C to G to C')
    print('Correction factor', len(X0X1new)/len(X0X1))


    torch.cuda.synchronize()
    ExtensiveDouble = ExtensiveDouble.to('cpu')
    IntensiveDouble = IntensiveDouble.to('cpu')
    start = time.time()
    ExtensiveDouble.to('cuda').to('cpu')
    IntensiveDouble.to('cuda').to('cpu')
    torch.cuda.synchronize()
    end = time.time()
    print('Elapsed', (end-start)*1000, 'ms for float64 transfering from C to G to C')

if __name__ == '__main__':
    test_special_numbers()
    test_perf()

    test_nb()

    try:
        fpath = sys.argv[1]
    except IndexError:
        exit(-1)
    plot_segments(fpath)
