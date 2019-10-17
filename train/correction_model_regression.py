import os
import sys

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K

sys.path.append('../preprocessing/')
from data_config import config as data_config

config = data_config()

X_train = np.load('../training_windows/correction_model_windows/X_train.npy')
Y_train = np.load('../training_windows/correction_model_windows/Y_train_regression.npy')
X_val = np.load('../training_windows/correction_model_windows/X_val.npy')
Y_val = np.load('../training_windows/correction_model_windows/Y_val_regression.npy')

print (X_train.shape)

m_train, w, f = X_train.shape
m_val, __, __  = X_val.shape
X_train_f = np.reshape(X_train, (m_train, w*f))
X_val_f = np.reshape(X_val, (m_val, w*f))

input_dim = w*f

model = Sequential([
    Dense(32, input_dim=input_dim),
    Activation('relu'),
    BatchNormalization(),
    Dense(20),
    Activation('relu'),
    Dense(5),
    Activation('relu'),
    Dense(1),
    Activation('tanh')
])

def avgabs(y_true,y_pred): ##in seconds
    return K.mean(K.abs(config.max_exp_perturbation*(y_true - y_pred)))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[avgabs])

model.fit(X_train_f, Y_train, epochs=25, validation_data=(X_val_f,Y_val),batch_size=32)

folder = '../models/correction_models/'
if not os.path.exists(folder):
    os.makedirs(folder)
model.save(folder + 'correction_model_reg.h5')