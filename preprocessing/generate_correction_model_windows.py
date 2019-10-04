import os
import random

import numpy as np
from sklearn.utils import shuffle

from data_config import config as data_config


X_train = []
Y_train_regression = []
Y_train_class = []
X_val = []
Y_val_regression = []
Y_val_class = []
X_test = []
Y_test_regression = []
Y_test_class = []

config = data_config()
samples_per_window = config.correction_window_s * config.fs

num_bins = int(2*config.fs*config.max_exp_perturbation)
bins = np.linspace(-1, 1, num=num_bins)
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
            digit_delta = np.digitize(scaled_delta, bins, right=True)
            class_delta = np.zeros((1, num_bins))
            class_delta[0, digit_delta] = 1
            x = X[window_center-int(samples_per_window/2):window_center+int(samples_per_window/2),:]
    #         print(x.shape)
            if x.shape != (samples_per_window, config.num_features):
                continue
            if i in config.train_files:
                X_train.append(x)
                Y_train_regression.append(scaled_delta)
                Y_train_class.append(class_delta)
            elif i in config.val_files:
                X_val.append(x)
                Y_val_regression.append(scaled_delta)
                Y_val_class.append(class_delta)
            elif i in config.test_files:
                X_test.append(x)
                Y_test_regression.append(scaled_delta)
                Y_test_class.append(class_delta)
                
X_train = np.stack(X_train)
Y_train_regression = np.reshape(np.stack(Y_train_regression), (len(Y_train_regression),1))
Y_train_class = np.reshape(np.stack(Y_train_class), (len(Y_train_class), num_bins))

X_val = np.stack(X_val)
Y_val_regression = np.reshape(np.stack(Y_val_regression), (len(Y_val_regression),1))
Y_val_class = np.reshape(np.stack(Y_val_class), (len(Y_val_class), num_bins))

X_test = np.stack(X_test)
Y_test_regression = np.reshape(np.stack(Y_test_regression), (len(Y_test_regression),1))
Y_test_class = np.reshape(np.stack(Y_test_class), (len(Y_test_class), num_bins))

X_train, Y_train_regression, Y_train_class = shuffle(X_train, 
                                                Y_train_regression,
                                                Y_train_class)

folder = "../training_windows/correction_model_windows/"
if not os.path.exists(folder):
    os.makedirs(folder)

np.save(folder + "X_train" , X_train)
np.save(folder + "Y_train_regression", Y_train_regression)
np.save(folder + "Y_train_class", Y_train_class)
np.save(folder + "X_val" , X_val)
np.save(folder + "Y_val_regression", Y_val_regression)
np.save(folder + "Y_val_class", Y_val_class)
np.save(folder + "X_test" , X_test)
np.save(folder + "Y_test_regression", Y_test_regression)
np.save(folder + "Y_test_class", Y_test_class)

