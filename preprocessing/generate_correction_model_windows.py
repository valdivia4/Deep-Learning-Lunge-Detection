import os
import random

import numpy as np
from sklearn.utils import shuffle

from data_config import config as data_config


X_train = []
Y_train_regression = []
X_val = []
Y_val_regression = []
X_test = []
Y_test_regression = []

config = data_config()
samples_per_window = config.correction_window_s * config.fs

for i in range(config.num_files):
    X = np.load('./numpy_data/inputs/inputs_'+ str(i)+'.npy')
    Y = np.load('./numpy_data/labels/labels_'+ str(i)+'.npy')
    
    indices = np.where(Y == 1)[0]
    
    for index in indices:
        for j in range(config.num_correction_windows_per_label):

            #### TODO: Possibility of different distribution tuned by user
            delta = np.random.normal()
            delta = round(delta*config.fs)
            window_center = index - delta
            scaled_delta = delta/(config.max_exp_perturbation*config.fs)
            x = X[window_center-int(samples_per_window/2):window_center+int(samples_per_window/2),:]
    #         print(x.shape)
            if x.shape != (samples_per_window, config.num_features):
                continue
            if i in config.train_files:
                X_train.append(x)
                Y_train_regression.append(scaled_delta)
            elif i in config.val_files:
                X_val.append(x)
                Y_val_regression.append(scaled_delta)
            elif i in config.test_files:
                X_test.append(x)
                Y_test_regression.append(scaled_delta)
                
X_train = np.stack(X_train)
Y_train_regression = np.reshape(np.stack(Y_train_regression), (len(Y_train_regression),1))

X_val = np.stack(X_val)
Y_val_regression = np.reshape(np.stack(Y_val_regression), (len(Y_val_regression),1))

X_test = np.stack(X_test)
Y_test_regression = np.reshape(np.stack(Y_test_regression), (len(Y_test_regression),1))

X_train, Y_train_regression = shuffle(X_train, Y_train_regression)
X_val, Y_val_regression = shuffle(X_val, Y_val_regression)
X_test, Y_test_regression = shuffle(X_test, Y_test_regression)

folder = "../training_windows/correction_model_windows/"
if not os.path.exists(folder):
    os.makedirs(folder)

np.save(folder + "X_train" , X_train)
np.save(folder + "Y_train_regression", Y_train_regression)
np.save(folder + "X_val" , X_val)
np.save(folder + "Y_val_regression", Y_val_regression)
np.save(folder + "X_test" , X_test)
np.save(folder + "Y_test_regression", Y_test_regression)

