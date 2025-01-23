from tred.loaders import StepLoader

import h5py
import matplotlib
import matplotlib.pyplot as plt

import torch
import numpy as np

import os
import sys

def plot_segments(fpath):
    f = h5py.File(fpath, 'r')
    segments = f['segments']
    steploader = StepLoader(f)
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


if __name__ == '__main__':
    try:
        fpath = sys.argv[1]
    except IndexError:
        exit(-1)
    plot_segments(fpath)
