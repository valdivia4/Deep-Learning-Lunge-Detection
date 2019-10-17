import os
import sys

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization

sys.path.append('../preprocessing/')
from data_config import config as data_config

#SET THIS:
num_epochs = 100

config = data_config()

X_train = np.load('../training_windows/correction_model_windows/X_train.npy')
Y_train = np.load('../training_windows/correction_model_windows/Y_train_class.npy')
X_val = np.load('../training_windows/correction_model_windows/X_val.npy')
Y_val = np.load('../training_windows/correction_model_windows/Y_val_class.npy')

m_train, w, f = X_train.shape
m_val, __, __  = X_val.shape
X_train_f = np.reshape(X_train, (m_train, w*f))
X_val_f = np.reshape(X_val, (m_val, w*f))

input_dim = w*f
num_bins = 2*config.fs*config.max_exp_perturbation

model = Sequential([
    Dense(32, input_dim=input_dim),
    BatchNormalization(),
    Activation('relu'),
    Dense(20),
    BatchNormalization(),
    Activation('relu'),
    Dense(5),
    BatchNormalization(),
    Activation('relu'),
    Dense(num_bins),
    Activation('softmax')
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['acc'])

model.fit(X_train_f, Y_train, epochs=num_epochs,
          validation_data=(X_val_f,Y_val), batch_size=32)

folder = '../models/correction_models/'
if not os.path.exists(folder):
    os.makedirs(folder)
model.save(folder + 'correction_model_class.h5')
