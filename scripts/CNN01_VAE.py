# general tools
import os
import sys
from glob import glob

# data tools
import re
import time
import h5py
import random
import numpy as np
from random import shuffle
from tensorflow import keras
from datetime import datetime, timedelta

#tf.config.run_functions_eagerly(True)

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

import tensorflow as tf

# ==================== #
weights_round = 1
save_round = 2
seeds = 123456
model_prefix_load = 'VAE_base{}'.format(weights_round) #False
model_prefix_save = 'VAE_base{}'.format(save_round)
N_vars = L_vars = 15
lr = 1e-4
# ==================== #
temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

vers = ['v3', 'v4x', 'v4'] # HRRR v4, v4x, v4
leads = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

filenames_pos_train = np.load(save_dir_campaign+'HRRR_filenames_pos_train.npy', allow_pickle=True)[()]
filenames_neg_train = np.load(save_dir_campaign+'HRRR_filenames_neg_train.npy', allow_pickle=True)[()]

# ------------------------------------------------------------------ #
# Merge train/valid and pos/neg batch files from multiple lead times
pos_train_all = []
neg_train_all = []

for ver in vers:
    for lead in leads:
        pos_train_all += filenames_pos_train['{}_lead{}'.format(ver, lead)]
        neg_train_all += filenames_neg_train['{}_lead{}'.format(ver, lead)]
        
filenames_train = pos_train_all + neg_train_all

ind_pick_from_batch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

with h5py.File(save_dir+'CNN_Validation_large.hdf', 'r') as h5io:
    VALID_input_64 = h5io['VALID_input_64'][...]
    VALID_target = h5io['VALID_target'][...]
    
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
latent_dim = 128
channels=[48, 64, 96, 128]

encoder_inputs = keras.Input(shape=(64, 64, 15))

X = encoder_inputs

X = keras.layers.Conv2D(channels[0], kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

X = keras.layers.Conv2D(channels[0], kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

# pooling
X = keras.layers.Conv2D(channels[1], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

X = keras.layers.Conv2D(channels[1], kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

X = keras.layers.Conv2D(channels[1], kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

# pooling
X = keras.layers.Conv2D(channels[2], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

X = keras.layers.Conv2D(channels[2], kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

X = keras.layers.Conv2D(channels[2], kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

# pooling
X = keras.layers.Conv2D(channels[3], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

X = keras.layers.Conv2D(channels[3], kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

X = keras.layers.Conv2D(channels[3], kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

V1 = X
OUT = keras.layers.GlobalMaxPooling2D()(V1)

z_mean = keras.layers.Dense(latent_dim, name="z_mean")(OUT)
z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(OUT)
z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

latent_inputs = keras.Input(shape=(latent_dim,))

X = latent_inputs

X = keras.layers.Dense(8*8*128, activation="gelu")(X)
X = keras.layers.Reshape((8, 8, 128))(X)

X = keras.layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), padding='same',)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

X = keras.layers.Conv2D(128, kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

# X = keras.layers.Conv2D(128, kernel_size=3, padding='same', use_bias=False)(X)
# X = keras.layers.BatchNormalization(axis=-1)(X)
# X = keras.layers.Activation("gelu")(X)

X = keras.layers.Conv2DTranspose(64, kernel_size=2, strides=(2, 2), padding='same',)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

X = keras.layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

# X = keras.layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False)(X)
# X = keras.layers.BatchNormalization(axis=-1)(X)
# X = keras.layers.Activation("gelu")(X)

X = keras.layers.Conv2DTranspose(32, kernel_size=2, strides=(2, 2), padding='same',)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

X = keras.layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)(X)
X = keras.layers.BatchNormalization(axis=-1)(X)
X = keras.layers.Activation("gelu")(X)

# X = keras.layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)(X)
# X = keras.layers.BatchNormalization(axis=-1)(X)
# X = keras.layers.Activation("gelu")(X)

OUT = keras.layers.Conv2D(15, kernel_size=1, padding='same', use_bias=True)(X)

decoder = keras.Model(latent_inputs, OUT, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_absolute_error(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
model_final = VAE(encoder, decoder)
model_final.compile(optimizer=keras.optimizers.Adam(lr=lr))

# load weights

model_name = '{}'.format(model_prefix_load)
model_path = temp_dir+model_name
model_final.load_weights(model_path+'hdf')

# ----- #

model_final.fit(VALID_input_64, batch_size=64, epochs=2000)

model_name = '{}'.format(model_prefix_save)
model_path = temp_dir+model_name
print('Saving weights')
model_final.save_weights(model_path+'hdf')

# ----- #

model_final.fit(VALID_input_64, batch_size=64, epochs=2000)

model_name = '{}'.format(model_prefix_save)
model_path = temp_dir+model_name
print('Saving weights')
model_final.save_weights(model_path+'hdf')

# ----- #

model_final.fit(VALID_input_64, batch_size=64, epochs=2000)

model_name = '{}'.format(model_prefix_save)
model_path = temp_dir+model_name
print('Saving weights')
model_final.save_weights(model_path+'hdf')

# ----- #

model_final.fit(VALID_input_64, batch_size=64, epochs=1000)

model_name = '{}'.format(model_prefix_save)
model_path = temp_dir+model_name
print('Saving weights')
model_final.save_weights(model_path+'hdf')

