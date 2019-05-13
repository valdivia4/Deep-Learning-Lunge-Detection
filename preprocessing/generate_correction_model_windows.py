import numpy as np
from sklearn.utils import shuffle
import random
from data_config import config as data_config


X_train = []
Y_train = []
X_val = []
Y_val = []
X_test = []
Y_test = []

config = data_config()
train = config.train_files
val = config.val_files
test = config.test_files

for i in range(config.num_files):
    X = np.load('./numpy_data/inputs/inputs_'+ str(i)+'.npy')
    Y = np.load('./numpy_data/labels/labels_'+ str(i)+'.npy')
    
    indices = np.where(Y == 1)[0]
    
    for index in indices:
        for j in range(num_windows_per_label):
            delta = random.randint(-perturbation_max,perturbation_max)
            window_center = index - delta
            scaled_delta = delta/(5*fs)
            x = X[window_center-int(WINDOW/2):window_center+int(WINDOW/2),:]
    #         print(x.shape)
            if i in train_files:
                X_train.append(x)
                Y_train.append(scaled_delta)
            elif i in val_files:
                X_val.append(x)
                Y_val.append(scaled_delta)
            elif i in test_files:
                X_test.append(x)
                Y_test.append(scaled_delta)
                
X_train = np.stack(X_train)
Y_train = np.reshape(np.stack(Y_train), (len(Y_train),1))

X_val = np.stack(X_val)
Y_val = np.reshape(np.stack(Y_val), (len(Y_val),1))

X_test = np.stack(X_test)
Y_test = np.reshape(np.stack(Y_test), (len(Y_test),1))

X_train, Y_train = shuffle(X_train, Y_train)
X_val, Y_val = shuffle(X_val, Y_val)
X_test, Y_test = shuffle(X_test, Y_test)

np.save("../training_windows/correction_model_windows/X_train" , X_train)
np.save("../training_windows/correction_model_windows/Y_train", Y_train)
np.save("../training_windows/correction_model_windows/X_val" , X_val)
np.save("../training_windows/correction_model_windows/Y_val", Y_val)
np.save("../training_windows/correction_model_windows/X_test" , X_test)
np.save("../training_windows/correction_model_windows/Y_test", Y_test)