import sys
from glob import glob

import time
import h5py
import zarr
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

from datetime import datetime, timedelta

#import dask.array as da

def fillnan(arr):
    '''
    fill NaNs with nearest neighbour grid point val
    The first grid point (left and bottom) cannot be NaNs
    output: grid
    '''
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out

def fill_coast_interp(arr, flag=False):
    '''
    Fill ocean grid points with the nearest land val
    sequence: left > top > right > bottom
    '''
    out = np.copy(arr) # copy
    # left fill
    out = np.fliplr(fillnan(np.fliplr(out)))
    # top fill
    out = np.rot90(fillnan(np.rot90(out, k=1)), k=3)
    # right fill
    out = fillnan(out)
    # bottom fill
    out = np.rot90(fillnan(np.rot90(out, k=3)), k=1)
    if type(flag) == bool:
        return out
    else:
        out[flag] = np.nan
        return out
    


with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_80km = h5io['lon_80km'][...]
    lat_80km = h5io['lat_80km'][...]
    land_mask_80km = h5io['land_mask_80km'][...]
    land_mask_3km = h5io['land_mask_3km'][...]

shape_3km = land_mask_3km.shape
shape_80km = land_mask_80km.shape

LEADs = [[2, 3, 4], [3, 4, 5], [4, 5, 6], 
         [5, 6, 7], [6, 7, 8], [7, 8, 9], 
         [8, 9, 10], [9, 10, 11], [10, 11, 12], 
         [11, 12, 13], [12, 13, 14], [13, 14, 15], 
         [14, 15, 16], [15, 16, 17], [16, 17, 18], 
         [17, 18, 19], [18, 19, 20], [19, 20, 21], 
         [20, 21, 22], [21, 22, 23], [22, 23, 24]]

N_vars = 15
shape_ = (65, 93, N_vars, len(LEADs))

mean_all = np.empty(shape_)
std_all = np.empty(shape_)
max_all = np.empty(shape_)

mean_all[...] = np.nan
std_all[...] = np.nan
max_all[...] = np.nan

for i, leads in enumerate(LEADs):
    norm_stats = np.load('/glade/work/ksha/NCAR/stats_allv4_80km_full_lead{}{}{}.npy'.format(leads[0], leads[1], leads[2]))
    max_stats = np.load('/glade/work/ksha/NCAR/p90_allv4_80km_full_lead{}{}{}.npy'.format(leads[0], leads[1], leads[2]))
    
    mean_temp = norm_stats[..., 0]
    std_temp = norm_stats[..., 1]
    max_temp = max_stats[..., -1]
    
    for n in range(N_vars):
        mean_all[..., n, i] = fill_coast_interp(mean_temp[..., n], flag=False)
        std_all[..., n, i]  = fill_coast_interp(std_temp[..., n], flag=False)
        max_all[..., n, i]  = fill_coast_interp(max_temp[..., n], flag=False)
        

save_dir = '/glade/work/ksha/NCAR/'

tuple_save = (mean_all, std_all, max_all)
label_save = ['mean_stats', 'std_stats', 'max_stats']
du.save_hdf5(tuple_save, label_save, save_dir, 'HRRRv4_STATS.hdf')
    
    
    
    
    