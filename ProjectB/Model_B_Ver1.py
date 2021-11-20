# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

from keras.layers import Lambda, Input, Dense, Reshape, RepeatVector, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.constraints import unit_norm, max_norm

from sklearn.impute import KNNImputer

from sklearn.manifold import MDS
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler



def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch,1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def drop_col(df, cutoff=0.2):
    n = len(df)
    cnt = df.count()
    cnt = cnt / n
    return df.loc[:, cnt[cnt >= cutoff].index]


df = pd.read_csv('../../projectB/train1/train1.csv', index_col=[0])
df = df.fillna(0)

training_feature = df.iloc[:, :-1].values
ground_truth_r = df.iloc[:, -1].values

scaler = StandardScaler()
scaler.fit(training_feature)
training_feature = scaler.transform(training_feature)

np.random.seed(seed=0)
original_dim = training_feature.shape[1]
num_train = training_feature.shape[0]

input_shape_x = (original_dim, )
input_shape_r = (1, )

intermediate_dim = 32
batch_size = 64
latent_dim = 8
epochs = 1000

inputs_r = Input(shape=input_shape_r, name='ground_truth')
inputs_x = Input(shape=input_shape_x, name='encoder_input')
# inputs_x_dropout = Dropout(0.25)(inputs_x)

inter_x1 = Dense(128, activation='tanh', name='encoder_intermediate')(inputs_x)
# inter_x1 = Dense(128, activation='tanh', name='encoder_intermediate')(inputs_x_dropout)
inter_x2 = Dense(intermediate_dim, activation='tanh', name='encoder_intermediate_2')(inter_x1)

r_mean = Dense(1, name='r_mean')(inter_x2)
r_log_var = Dense(1, name='r_log_var')(inter_x2)

z_mean = Dense(latent_dim, name='z_mean')(inter_x2)
z_log_var = Dense(latent_dim, name='z_log_var')(inter_x2)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
r = Lambda(sampling, output_shape=(1,), name='r')([r_mean, r_log_var])

pz_mean = Dense(latent_dim, name='pz_mean',kernel_constraint=unit_norm())(r)

encoder = Model([inputs_x,inputs_r], [z_mean, z_log_var, z, r_mean, r_log_var, r, pz_mean], name='encodr')

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
inter_y1 = Dense(intermediate_dim, activation='tanh')(latent_inputs)
inter_y2 = Dense(128, activation='tanh')(inter_y1)
outputs = Dense(original_dim)(inter_y2)

decoder = Model(latent_inputs, outputs, name='decoder')

outputs = decoder(encoder([inputs_x,inputs_r])[2])
vae = Model([inputs_x,inputs_r], outputs, name='vae_mlp')

models = (encoder, decoder)

reconstruction_loss = mse(inputs_x,outputs)
kl_loss = 1 + z_log_var - K.square(z_mean-pz_mean) - K.exp(z_log_var)
kl_loss = -0.5*K.sum(kl_loss, axis=-1)

# label_loss = 1 + r_log_var - K.square(r_mean-inputs_r) - K.exp(r_log_var)
# label_loss = -0.5*K.sum(label_loss, axis=-1)
# label_loss = mse(r, inputs_r)

label_loss = K.tf.divide(0.5*K.square(r_mean-inputs_r),K.exp(r_log_var))+0.5*r_log_var

vae_loss = K.mean(reconstruction_loss+kl_loss+label_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.save_weights('./weight/random_weights.h5')

np.random.seed(0)
skf = StratifiedKFold(n_splits=10)
pred = np.zeros((ground_truth_r.shape))
fake = np.zeros((ground_truth_r.shape[0]))
fake[:300] = 1

for train_idx, test_idx in skf.split(training_feature,fake):
    training_feature_sk = training_feature[train_idx,:]
    training_score = ground_truth_r[train_idx]

    testing_feature_sk = training_feature[test_idx,:]
    testing_score = ground_truth_r[test_idx]

    filepath="./weight/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False,mode='min')
    callbacks_list = [checkpoint]

    vae.fit([training_feature_sk,training_score],
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1)
    [z_mean, z_log_var, z, r_mean, r_log_var, r_vae, pz_mean] = encoder.predict([testing_feature_sk,tesing_score],batch_size=batch_size)
    pred[test_idx] = r_mean[:,0]

# The mean squared error
print("Mean squared error: %.3f" % mean_squared_error(ground_truth_r, pred))
# Explained variance score: 1 is perfect prediction
print('R2 Variance score: %.3f' % r2_score(ground_truth_r, pred))
