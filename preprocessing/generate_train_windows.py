#!/usr/bin/env python
# coding: utf-8
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import math
from data_config import config as data_config

config = data_config()
num_train_sets = config.num_train_sets
species_code = config.species_code
window_s = config.window_s

## near_pos_multiple is the multiple of near positive training examples to keep relative to positive training examples
## random_neg_multiple is the multiple of randomly selected negative examples

near_pos_multiple = config.near_pos_multiple ##make sure fs*multiple is an integer, usually fs=10
rand_neg_multiple = config.rand_neg_multiple
   
num_features = config.num_features
SAMPLES_PER_S = config.SAMPLES_PER_S
padded_window_s = config.padded_window_s

num_files = config.num_files
WINDOW = window_s*SAMPLES_PER_S
PADDING = int((padded_window_s-window_s)/2) * SAMPLES_PER_S
    
PADDED_WINDOW = 2*PADDING+WINDOW
pos_keep = WINDOW
near_pos_keep = int(near_pos_multiple*pos_keep)
np.random.seed(8)
files = list(range(num_files))

train_files = config.train_files
val_files = config.val_files
test_files = config.test_files

print ('test: ' , test_files)
print('val: ',val_files)
print('train: ', train_files)

#Gets positive and negative samples for the deployment with features X and labels Y
#pos_keep positive samples are returned for every positive label
#near_pos_keep = near_pos_multiple*pos_keep nearly positive samples are kept as well as 
#rand_neg_keep = rand_neg_multiple*pos_keep randomly sampled negative samples
def get_pos_and_neg_samples(X, Y, PADDED_WINDOW, pos_keep, near_pos_keep, rand_neg_multiple):
    seed=6
    np.random.seed(seed)
    m, n = X.shape
    indices = np.where(Y == 1)[0]
    #get rid of indices whose window will extend beyond the deployment
    indices = [ind for ind in indices if ind < m-PADDED_WINDOW]

    pos_samples = set([])
    near_pos_samples = set([])
    for ind in indices:
        pos_indices = [ind-w for w in range(PADDING, PADDING+WINDOW + 1)]
        pos_indices = np.random.choice(pos_indices, pos_keep,replace=False)
        count = 0
        for k in pos_indices:
            pos_samples.add(k)
            count+=1

        #choose near_pos_keep near positive indices per positive example
        diff = list(range(PADDING))+ list(range(PADDING+WINDOW + 1, 2*PADDING+WINDOW+1))
        near_pos_indices = [ind-w for w in diff]
        near_pos_indices = np.random.choice(near_pos_indices, near_pos_keep,replace=False)
        for k in near_pos_indices:
            near_pos_samples.add(k)
                
    #neg times that haven't been selected
    unselected_neg_samples = (set(range(0,m-PADDED_WINDOW)) - pos_samples) - near_pos_samples
    rand_neg_keep = int(rand_neg_multiple*len(pos_samples))
    neg_samples = np.random.choice(list(unselected_neg_samples), min(rand_neg_keep, len(unselected_neg_samples)), replace=False)
    neg_samples = list(neg_samples)
    neg_samples.extend(list(near_pos_samples))
    pos_samples = list(pos_samples)
    return pos_samples, neg_samples

def get_train_set_size():
    num_train_pos, num_train_neg = 0, 0
    for i in train_files:
        X = np.load('./numpy_data/inputs/inputs_'+ str(i)+'.npy')
        Y = np.load('./numpy_data/labels/labels_'+ str(i)+'.npy')
        pos_samples, neg_samples = get_pos_and_neg_samples(X, Y, PADDED_WINDOW, pos_keep, near_pos_keep, rand_neg_multiple)
        num_pos, num_neg = len(pos_samples), len(neg_samples)
       
        num_train_pos += num_pos
        num_train_neg +=num_neg 
    num_train = num_train_pos + num_train_neg
    return num_train_pos, num_train_neg, num_train

# Find out how big training data is:
num_train_pos, num_train_neg, num_train = get_train_set_size()
print ('Positive Training Examples = ',int(num_train_pos))
print ('Negative Training Examples = ' ,int(num_train_neg))
print ('Total = ', num_train)
#Creates and saves the training set windows and trainval set windows
#based on the given parameters

seen = set([])
def saveTrainValSet(near_pos_multiple,rand_neg_multiple, WINDOW, PADDED_WINDOW, 
                 num_files,train_files, val_files, num_train,species_code, pathname = ''):
    
   #size of each training set block 
    train_block_size = int(math.ceil(num_train/num_train_sets))
    train_shuffle = np.random.permutation(num_train)
    
    ### make num_train_sets train sets indexed by j
    for j in range(num_train_sets):
        X_train = np.zeros((train_block_size,PADDED_WINDOW, num_features))
        Y_train = np.zeros((train_block_size,1))

        train_index = 0
        for i in train_files:
            X = np.load('./numpy_data/inputs/inputs_'+ str(i)+'.npy')
            Y = np.load('./numpy_data/labels/labels_'+ str(i)+'.npy')
            pos_samples, neg_samples = get_pos_and_neg_samples(X, Y, PADDED_WINDOW, pos_keep, near_pos_keep, rand_neg_multiple)
            # print(len(pos_samples))
            for index in pos_samples:
                x = X[index:index+PADDED_WINDOW,:]
                y = 1
                shuff_ind = train_shuffle[train_index]
                # print(shuff_ind, j, train_set_size)
                train_index +=1
                if shuff_ind >= j*train_block_size and shuff_ind < (j+1)*train_block_size:
                    assert not (np.any(np.isnan(x)))
                    seen.add(shuff_ind)
                    train_ind = shuff_ind - j*train_block_size
                    X_train[train_ind,:] = x
                    Y_train[train_ind] = y
                    # print("This")
            for index in neg_samples:
                x = X[index:index+PADDED_WINDOW,:]
                y = 0
                assert not (np.any(np.isnan(x)))
                shuff_ind = train_shuffle[train_index]
                train_index +=1
                if shuff_ind >= j*train_block_size and shuff_ind < (j+1)*train_block_size:
                    seen.add(shuff_ind)
                    train_ind = shuff_ind - j*train_block_size
                    X_train[train_ind,:] = x
                    Y_train[train_ind] = y
        lwpath_xtrain = "X_train_npm_{}_rnm_{}_num_{}".format(near_pos_multiple, rand_neg_multiple, j)

        np.save("../training_windows/label_model_windows/" + lwpath_xtrain, X_train)

        lwpath_ytrain = "Y_train_npm_{}_rnm_{}_num_{}".format(near_pos_multiple, rand_neg_multiple, j)
        np.save("../training_windows/label_model_windows/" + lwpath_ytrain, Y_train)

        print(train_block_size, X_train.shape, Y_train.shape)
        X_train, Y_train = None, None

        assert(train_index == num_train)

    print (len(seen))
    X_trainval = []
    Y_trainval = []       
     
        
    ############ Make the trainval set    
    for i in val_files:
        X = np.load('./numpy_data/inputs/inputs_'+ str(i)+'.npy')
        Y = np.load('./numpy_data/labels/labels_'+ str(i)+'.npy')
        
        pos_samples, neg_samples =  get_pos_and_neg_samples(X, Y, PADDED_WINDOW, pos_keep, near_pos_keep, rand_neg_multiple)
        for index in pos_samples:
            x = X[index:index+PADDED_WINDOW,:]
    #         print(x.shape)
            y = 1
            
            X_trainval.append(x)
            Y_trainval.append(y)
        for index in neg_samples:
            x = X[index:index+PADDED_WINDOW,:]
            y = 0
            X_trainval.append(x)
            Y_trainval.append(y)
    X_trainval = np.stack(X_trainval)
    Y_trainval = np.reshape(np.stack(Y_trainval), (len(Y_trainval),1))


    # X_trainval, Y_trainval = shuffle(X_trainval, Y_trainval)


    print(X_trainval.shape, Y_trainval.shape)
    np.save("../training_windows/label_model_windows/" + "X_trainval_npm_{}_rnm_{}.npy".format(near_pos_multiple, rand_neg_multiple)
, X_trainval)
    np.save("../training_windows/label_model_windows/" + "Y_trainval_npm_{}_rnm_{}.npy".format(near_pos_multiple, rand_neg_multiple)
, Y_trainval)

saveTrainValSet(near_pos_multiple,rand_neg_multiple, WINDOW, PADDED_WINDOW, 
                 num_files, train_files, val_files, num_train,species_code, pathname = '')

