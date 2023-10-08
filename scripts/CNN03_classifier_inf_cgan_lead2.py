'''
Train classifier head with size-128 feature vectors on a 64-by-64 grid cell
Nearby grid cells are not considered
Revamped
'''

# general tools
import os
import re
import sys
import time
import h5py
import random
import scipy.ndimage
from glob import glob

import numpy as np
from datetime import datetime, timedelta
from random import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from keras_unet_collection import utils as k_utils

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

def feature_extract(filenames, lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max):
    
    lon_out = []
    lat_out = []
    elev_out = []

    for i, name in enumerate(filenames):
        
        nums = re.findall(r'\d+', name)
        indy = int(nums[-2])
        indx = int(nums[-3])
        
        lon = lon_80km[indx, indy]
        lat = lat_80km[indx, indy]

        lon = (lon - lon_minmax[0])/(lon_minmax[1] - lon_minmax[0])
        lat = (lat - lat_minmax[0])/(lat_minmax[1] - lat_minmax[0])

        elev = elev_80km[indx, indy]
        elev = elev / elev_max
        
        lon_out.append(lon)
        lat_out.append(lat)
        elev_out.append(elev)
        
    return np.array(lon_out), np.array(lat_out), np.array(elev_out)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead1', help='lead1')
parser.add_argument('lead2', help='lead2')
parser.add_argument('lead3', help='lead3')
parser.add_argument('lead4', help='lead4')

args = vars(parser.parse_args())

# =============== #

lead1 = int(args['lead1'])
lead2 = int(args['lead2'])
lead3_ = int(args['lead3'])
lead4_ = int(args['lead4'])

lead_name = lead1
model_tag = 'vgg2' #args['model_tag']

L_vec = 2
N = 50

# ================================================================ #
# Geographical information

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_80km = h5io['lon_80km'][...]
    lat_80km = h5io['lat_80km'][...]
    elev_3km = h5io['elev_3km'][...]
    land_mask_80km = h5io['land_mask_80km'][...]
    
grid_shape = land_mask_80km.shape

elev_80km = du.interp2d_wraper(lon_3km, lat_3km, elev_3km, lon_80km, lat_80km, method='linear')

elev_80km[np.isnan(elev_80km)] = 0
elev_80km[elev_80km<0] = 0
elev_max = np.max(elev_80km)

lon_80km_mask = lon_80km[land_mask_80km]
lat_80km_mask = lat_80km[land_mask_80km]

lon_minmax = [np.min(lon_80km_mask), np.max(lon_80km_mask)]
lat_minmax = [np.min(lat_80km_mask), np.max(lat_80km_mask)]

# ============================================================ #
# File path
path_name1_v3 = path_batch_v3
path_name2_v3 = path_batch_v3
path_name3_v3 = path_batch_v3
path_name4_v3 = path_batch_v3

path_name1_v4 = path_batch_v4x
path_name2_v4 = path_batch_v4x
path_name3_v4 = path_batch_v4x
path_name4_v4 = path_batch_v4x

path_name1_v4_test = path_batch_v4
path_name2_v4_test = path_batch_v4
path_name3_v4_test = path_batch_v4
path_name4_v4_test = path_batch_v4

temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

# filenames_train_v3 = np.load('/glade/work/ksha/NCAR/filenames_for_TRAIN_inds.npy', allow_pickle=True)[()]
# filenames_valid_v3 = np.load('/glade/work/ksha/NCAR/filenames_for_VALID_inds.npy', allow_pickle=True)[()]

# filenames_train_v4x = np.load('/glade/work/ksha/NCAR/filenames_for_TRAIN_inds_v4x.npy', allow_pickle=True)[()]
# filenames_valid_v4x = np.load('/glade/work/ksha/NCAR/filenames_for_VALID_inds_v4x.npy', allow_pickle=True)[()]

# filename_train_lead1_v3 = filenames_train_v3['lead{}'.format(lead1)]
# filename_train_lead2_v3 = filenames_train_v3['lead{}'.format(lead2)]

# filename_valid_lead1_v3 = filenames_valid_v3['lead{}'.format(lead1)]
# filename_valid_lead2_v3 = filenames_valid_v3['lead{}'.format(lead2)]

# filename_train_lead1_v4x = filenames_train_v4x['lead{}'.format(lead1)]
# filename_train_lead2_v4x = filenames_train_v4x['lead{}'.format(lead2)]

# filename_valid_lead1_v4x = filenames_valid_v4x['lead{}'.format(lead1)]
# filename_valid_lead2_v4x = filenames_valid_v4x['lead{}'.format(lead2)]

# # ============================================================ #
# # Consistency check indices

# IND_TRAIN_lead = np.load('/glade/work/ksha/NCAR/IND_TRAIN_lead_{}{}{}{}.npy'.format(lead1, lead2, lead3_, lead4_), allow_pickle=True)[()]
# TRAIN_ind1_v3 = IND_TRAIN_lead['lead{}'.format(lead1)]
# TRAIN_ind2_v3 = IND_TRAIN_lead['lead{}'.format(lead2)]
# TRAIN_ind3_v3 = IND_TRAIN_lead['lead{}'.format(lead3_)]
# TRAIN_ind4_v3 = IND_TRAIN_lead['lead{}'.format(lead4_)]

# IND_VALID_lead = np.load('/glade/work/ksha/NCAR/IND_VALID_lead_{}{}{}{}.npy'.format(lead1, lead2, lead3_, lead4_), allow_pickle=True)[()]
# VALID_ind1_v3 = IND_VALID_lead['lead{}'.format(lead1)]
# VALID_ind2_v3 = IND_VALID_lead['lead{}'.format(lead2)]
# VALID_ind3_v3 = IND_VALID_lead['lead{}'.format(lead3_)]
# VALID_ind4_v3 = IND_VALID_lead['lead{}'.format(lead4_)]

# IND_TRAIN_lead = np.load('/glade/work/ksha/NCAR/IND_TRAIN_lead_{}{}{}{}_v4x.npy'.format(lead1, lead2, lead3_, lead4_), allow_pickle=True)[()]
# TRAIN_ind1_v4x = IND_TRAIN_lead['lead{}'.format(lead1)]
# TRAIN_ind2_v4x = IND_TRAIN_lead['lead{}'.format(lead2)]
# TRAIN_ind3_v4x = IND_TRAIN_lead['lead{}'.format(lead3_)]
# TRAIN_ind4_v4x = IND_TRAIN_lead['lead{}'.format(lead4_)]

# IND_VALID_lead = np.load('/glade/work/ksha/NCAR/IND_VALID_lead_{}{}{}{}_v4x.npy'.format(lead1, lead2, lead3_, lead4_), allow_pickle=True)[()]
# VALID_ind1_v4x = IND_VALID_lead['lead{}'.format(lead1)]
# VALID_ind2_v4x = IND_VALID_lead['lead{}'.format(lead2)]
# VALID_ind3_v4x = IND_VALID_lead['lead{}'.format(lead3_)]
# VALID_ind4_v4x = IND_VALID_lead['lead{}'.format(lead4_)]

# data_lead1_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
# data_lead1_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
# data_lead1_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
# data_lead1_p3 = np.load('{}TRAIN_v3_vec_lead{}_part3_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]

# data_lead2_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
# data_lead2_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
# data_lead2_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
# data_lead2_p3 = np.load('{}TRAIN_v3_vec_lead{}_part3_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]

# TRAIN_lead1_v3 = np.concatenate((data_lead1_p0['y_vector'], data_lead1_p1['y_vector'], data_lead1_p2['y_vector'], data_lead1_p3['y_vector']), axis=0)
# TRAIN_lead2_v3 = np.concatenate((data_lead2_p0['y_vector'], data_lead2_p1['y_vector'], data_lead2_p2['y_vector'], data_lead2_p3['y_vector']), axis=0)

# TRAIN_lead1_y_v3 = np.concatenate((data_lead1_p0['y_true'], data_lead1_p1['y_true'], data_lead1_p2['y_true'], data_lead1_p3['y_true']), axis=0)
# TRAIN_lead2_y_v3 = np.concatenate((data_lead2_p0['y_true'], data_lead2_p1['y_true'], data_lead2_p2['y_true'], data_lead2_p3['y_true']), axis=0)

# # =========================================================== #
# # Load feature vectors (HRRR v3, validation)

# data_lead1_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
# data_lead2_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]

# VALID_lead1_v3 = data_lead1_valid['y_vector']
# VALID_lead2_v3 = data_lead2_valid['y_vector']

# VALID_lead1_y_v3 = data_lead1_valid['y_true']
# VALID_lead2_y_v3 = data_lead2_valid['y_true']

# data_lead1_p0 = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
# data_lead2_p0 = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]

# TRAIN_lead1_v4x = data_lead1_p0['y_vector']
# TRAIN_lead2_v4x = data_lead2_p0['y_vector']

# TRAIN_lead1_y_v4x = data_lead1_p0['y_true']
# TRAIN_lead2_y_v4x = data_lead2_p0['y_true']

# # =========================================================== #
# # Load feature vectors (HRRR v4x, validation)

# data_lead1_valid = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
# data_lead2_valid = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]

# VALID_lead1_v4x = data_lead1_valid['y_vector']
# VALID_lead2_v4x = data_lead2_valid['y_vector']

# VALID_lead1_y_v4x = data_lead1_valid['y_true']
# VALID_lead2_y_v4x = data_lead2_valid['y_true']

# # ================================================================= #
# # Collect feature vectors from all batch files (HRRR v3, validation)

# L = len(TRAIN_ind2_v3)

# filename_train1_pick_v3 = []
# filename_train2_pick_v3 = []

# TRAIN_v3_lead1 = np.empty((L, 128))
# TRAIN_v3_lead2 = np.empty((L, 128))

# TRAIN_Y_v3 = np.empty(L)

# for i in range(L):
    
#     ind_lead1_v3 = int(TRAIN_ind1_v3[i])
#     ind_lead2_v3 = int(TRAIN_ind2_v3[i])
    
#     filename_train1_pick_v3.append(filename_train_lead1_v3[ind_lead1_v3])
#     filename_train2_pick_v3.append(filename_train_lead2_v3[ind_lead2_v3])

#     TRAIN_v3_lead1[i, :] = TRAIN_lead1_v3[ind_lead1_v3, :]
#     TRAIN_v3_lead2[i, :] = TRAIN_lead2_v3[ind_lead2_v3, :]
    
#     TRAIN_Y_v3[i] = TRAIN_lead1_y_v3[ind_lead1_v3]

# # ================================================================= #
# # Collect feature vectors from all batch files (HRRR v4x, validation)

# L = len(TRAIN_ind2_v4x)

# filename_train1_pick_v4x = []
# filename_train2_pick_v4x = []

# TRAIN_v4x_lead1 = np.empty((L, 128))
# TRAIN_v4x_lead2 = np.empty((L, 128))

# TRAIN_Y_v4x = np.empty(L)

# for i in range(L):
    
#     ind_lead1_v4x = int(TRAIN_ind1_v4x[i])
#     ind_lead2_v4x = int(TRAIN_ind2_v4x[i])

#     filename_train1_pick_v4x.append(filename_train_lead1_v4x[ind_lead1_v4x])
#     filename_train2_pick_v4x.append(filename_train_lead2_v4x[ind_lead2_v4x])

#     TRAIN_v4x_lead1[i, :] = TRAIN_lead1_v4x[ind_lead1_v4x, :]
#     TRAIN_v4x_lead2[i, :] = TRAIN_lead2_v4x[ind_lead2_v4x, :]

#     TRAIN_Y_v4x[i] = TRAIN_lead1_y_v4x[ind_lead1_v4x]

# # ================================================================== #
# # Collect feature vectors from all batch files (HRRR v3, validation)
# L = len(VALID_ind2_v3)

# filename_valid1_pick_v3 = []
# filename_valid2_pick_v3 = []

# VALID_v3_lead1 = np.empty((L, 128))
# VALID_v3_lead2 = np.empty((L, 128))

# VALID_Y_v3 = np.empty(L)

# for i in range(L):
    
#     ind_lead1_v3 = int(VALID_ind1_v3[i])
#     ind_lead2_v3 = int(VALID_ind2_v3[i])

#     filename_valid1_pick_v3.append(filename_valid_lead1_v3[ind_lead1_v3])
#     filename_valid2_pick_v3.append(filename_valid_lead2_v3[ind_lead2_v3])

#     VALID_v3_lead1[i, :] = VALID_lead1_v3[ind_lead1_v3, :]
#     VALID_v3_lead2[i, :] = VALID_lead2_v3[ind_lead2_v3, :]

#     VALID_Y_v3[i] = VALID_lead1_y_v3[ind_lead1_v3]

# L = len(VALID_ind2_v4x)

# filename_valid1_pick_v4x = []
# filename_valid2_pick_v4x = []

# VALID_v4x_lead1 = np.empty((L, 128))
# VALID_v4x_lead2 = np.empty((L, 128))

# VALID_Y_v4x = np.empty(L)

# for i in range(L):
    
#     ind_lead1_v4x = int(VALID_ind1_v4x[i])
#     ind_lead2_v4x = int(VALID_ind2_v4x[i])

#     filename_valid1_pick_v4x.append(filename_valid_lead1_v4x[ind_lead1_v4x])
#     filename_valid2_pick_v4x.append(filename_valid_lead2_v4x[ind_lead2_v4x])

#     VALID_v4x_lead1[i, :] = VALID_lead1_v4x[ind_lead1_v4x, :]
#     VALID_v4x_lead2[i, :] = VALID_lead2_v4x[ind_lead2_v4x, :]

#     VALID_Y_v4x[i] = VALID_lead1_y_v4x[ind_lead1_v4x]

# #VALID_Y_v4x_adjust = VALID_Y_v4x

# ALL_TRAIN_v3 = np.concatenate((TRAIN_v3_lead1[:, None, :], TRAIN_v3_lead2[:, None, :]), axis=1)
# ALL_VALID_v3 = np.concatenate((VALID_v3_lead1[:, None, :], VALID_v3_lead2[:, None, :]), axis=1)

# ALL_TRAIN_v4x = np.concatenate((TRAIN_v4x_lead1[:, None, :], TRAIN_v4x_lead2[:, None, :]), axis=1)
# ALL_VALID_v4x = np.concatenate((VALID_v4x_lead1[:, None, :], VALID_v4x_lead2[:, None, :]), axis=1)

# ALL_VEC = np.concatenate((ALL_TRAIN_v3, ALL_VALID_v3, ALL_TRAIN_v4x, ALL_VALID_v4x), axis=0)
# TRAIN_Y = np.concatenate((TRAIN_Y_v3, VALID_Y_v3, TRAIN_Y_v4x, VALID_Y_v4x), axis=0)

# lon_norm, lat_norm, elev_norm = feature_extract(filename_train1_pick_v3+filename_train1_pick_v4x, 
#                                                 lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)
# TRAIN_stn = np.concatenate((lon_norm[:, None], lat_norm[:, None]), axis=1)

# lon_norm, lat_norm, elev_norm = feature_extract(filename_valid1_pick_v3+filename_valid1_pick_v4x, 
#                                                 lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)
# VALID_stn = np.concatenate((lon_norm[:, None], lat_norm[:, None]), axis=1)

# ALL_stn = np.concatenate((TRAIN_stn, VALID_stn))


# ====================================================== #
# HRRR v4x validation set
# ====================================================== #
# Read batch file names (npy)

filename_valid_lead1 = sorted(glob("{}TEST*lead{}.npy".format(path_name1_v4_test, lead1)))
filename_valid_lead2 = sorted(glob("{}TEST*lead{}.npy".format(path_name2_v4_test, lead2)))

# =============================== #
# Load feature vectors

valid_lead1 = np.load('{}TEST_v4_vec_lead{}_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
valid_lead2 = np.load('{}TEST_v4_vec_lead{}_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]

VALID_lead1 = valid_lead1['y_vector']
VALID_lead2 = valid_lead2['y_vector']

VALID_lead1_y = valid_lead1['y_true']
VALID_lead2_y = valid_lead2['y_true']

# ============================================================ #
# Consistency check indices

IND_TEST_lead = np.load('/glade/work/ksha/NCAR/IND_TEST_lead_v4.npy', allow_pickle=True)[()]

VALID_ind1 = IND_TEST_lead['lead{}'.format(lead1)]
VALID_ind2 = IND_TEST_lead['lead{}'.format(lead2)]

# ================================================================== #
# Collect feature vectors from all batch files

L = len(VALID_ind2)

filename_valid1_pick = []
filename_valid2_pick = []

VALID_X_lead1 = np.empty((L, 128))
VALID_X_lead2 = np.empty((L, 128))

VALID_Y = np.empty(L)

for i in range(L):
    
    ind_lead1 = int(VALID_ind1[i])
    ind_lead2 = int(VALID_ind2[i])

    filename_valid1_pick.append(filename_valid_lead1[ind_lead1])
    filename_valid2_pick.append(filename_valid_lead2[ind_lead2])

    VALID_X_lead1[i, :] = VALID_lead1[ind_lead1, :]
    VALID_X_lead2[i, :] = VALID_lead2[ind_lead2, :]

    VALID_Y[i] = VALID_lead1_y[ind_lead1]
    
VALID_VEC = np.concatenate((VALID_X_lead1[:, None, :], VALID_X_lead2[:, None, :]), axis=1)

# ================================================================== #
# extract location information

lon_norm, lat_norm, elev_norm = feature_extract(filename_valid1_pick, lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)
VALID_stn = np.concatenate((lon_norm[:, None], lat_norm[:, None]), axis=1)

# ================================================================================ #
# Inference

model = mu.create_classif_head(L_vec)
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=keras.optimizers.Adam(lr=0))

Y_pred_test_ens = np.empty((len(VALID_Y), N)); Y_pred_test_ens[...] = np.nan
# Y_pred_train_ens = np.empty((len(ALL_VEC), N)); Y_pred_train_ens[...] = np.nan

valid_shape = VALID_VEC.shape
#train_shape = ALL_VEC.shape

for n in range(N):

    print("round {}".format(n))

    W_old = k_utils.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}_lead{}_mc{}'.format(model_tag, lead_name, n))
    model.set_weights(W_old)
    
    Y_pred_test_ens[:, n] = model.predict([VALID_VEC, VALID_stn])[:, 0]
    # Y_pred_train_ens[:, n] = model.predict([ALL_VEC, ALL_stn])[:, 0]
    
# Save results
save_dict = {}
save_dict['Y_pred_test_ens'] = Y_pred_test_ens
save_dict['VALID_Y'] = VALID_Y

# save_dict['Y_pred_train_ens'] = Y_pred_train_ens
# save_dict['TRAIN_Y'] = TRAIN_Y

np.save('{}RESULT_cgan_lead{}_{}.npy'.format(filepath_vec, lead_name, model_tag), save_dict)
print('{}RESULT_cgan_lead{}_{}.npy'.format(filepath_vec, lead_name, model_tag))

