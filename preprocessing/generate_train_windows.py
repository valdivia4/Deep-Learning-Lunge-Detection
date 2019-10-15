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

#read config variables
num_train_sets = config.num_train_sets
window_s = config.window_s
near_pos_multiple = config.near_pos_multiple 
rand_neg_multiple = config.rand_neg_multiple
num_features = config.num_features
fs = config.fs
padded_window_s = config.padded_window_s
num_files = config.num_files
train_files = config.train_files
val_files = config.val_files
test_files = config.test_files

#get variables in samples instead of seconds
window = window_s*fs
padding = int((padded_window_s-window_s)/2) * fs
padded_window = 2*padding+window

pos_keep = window
near_pos_keep = int(near_pos_multiple*pos_keep)
files = list(range(num_files))

print ('test: ' , test_files)
print('val: ',val_files)
print('train: ', train_files)


def is_valid_index(index, padded_window, m):
    #returns True if the window [index,index+padded_window) is in the valid range [0, m) 
    return (index < m-padded_window) and (index>= 0)

def get_pos_and_neg_samples(X, Y, padded_window, pos_keep, near_pos_keep, rand_neg_multiple):
    #Gets positive and negative samples for the deployment with features X and labels Y
    #pos_keep positive samples are returned for every positive label
    #near_pos_keep = near_pos_multiple*pos_keep nearly positive samples are kept as well as 
    #rand_neg_keep = rand_neg_multiple*pos_keep randomly sampled negative samples

    m, n = X.shape
    indices = np.where(Y == 1)[0]

    #get positive samples
    pos_samples = set([])
    near_pos_samples = set([])
    for ind in indices:
        pos_indices = [ind-w for w in range(padding, padding+window + 1)]
        pos_indices = np.random.choice(pos_indices, pos_keep,replace=False)
        count = 0
        for k in pos_indices:
            if is_valid_index(k, padded_window, m):
                pos_samples.add(k)
                count+=1

        #choose near_pos_keep near positive indices per positive example
        diff = list(range(padding))+ list(range(padding+window + 1, 
                2*padding+window+1))
        near_pos_indices = [ind-w for w in diff]
        near_pos_indices = np.random.choice(
            near_pos_indices, near_pos_keep, replace=False
        )
        for k in near_pos_indices:
            if is_valid_index(k, padded_window, m):
                near_pos_samples.add(k)
                
    #neg times that haven't been selected
    unselected_neg_samples = set(
            [ind for ind in range(0,m-padded_window) 
                if is_valid_index(ind, padded_window, m)]
    )
    unselected_neg_samples = ((unselected_neg_samples - pos_samples) 
                                - near_pos_samples)
    rand_neg_keep = int(rand_neg_multiple*len(pos_samples))
    neg_samples = np.random.choice(list(unselected_neg_samples), 
            min(rand_neg_keep, len(unselected_neg_samples)), replace=False)
    neg_samples = list(neg_samples)
    neg_samples.extend(list(near_pos_samples))
    pos_samples = list(pos_samples)
    return pos_samples, neg_samples

def get_train_set_size():
    num_train_pos, num_train_neg = 0, 0
    for i in train_files:
        X = np.load('./numpy_data/inputs/inputs_'+ str(i)+'.npy')
        Y = np.load('./numpy_data/labels/labels_'+ str(i)+'.npy')
        pos_samples, neg_samples = get_pos_and_neg_samples(
                X, Y, padded_window, pos_keep, near_pos_keep, rand_neg_multiple
        )
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

def in_current_block(n, block_number, block_size):
    '''Returns whether the index n is in block block_number'''
    return (n >= block_number*block_size
        and n < (block_number+1)*block_size)

def save_train_set(near_pos_multiple,rand_neg_multiple, window, padded_window, 
                 num_files, train_files, val_files, num_train, pathname = ''):
    #Creates and saves the training set windows and trainval set windows
    #based on the given parameters
    
    #size of each training set block 
    train_block_size = int(math.ceil(num_train/num_train_sets))
    train_shuffle = np.random.permutation(num_train)
    
    #make num_train_sets train sets indexed by j
    for j in range(num_train_sets):
        X_train = np.zeros((train_block_size,padded_window, num_features))
        Y_train = np.zeros((train_block_size,1))

        train_index = 0
        for i in train_files:
            X = np.load('./numpy_data/inputs/inputs_'+ str(i)+'.npy')
            Y = np.load('./numpy_data/labels/labels_'+ str(i)+'.npy')
            pos_samples, neg_samples = get_pos_and_neg_samples(X, Y, 
                    padded_window, pos_keep, near_pos_keep, rand_neg_multiple
            )
            for index in pos_samples:
                x = X[index:index+padded_window,:]
                y = 1
                shuff_ind = train_shuffle[train_index]
                train_index +=1
                if in_current_block(shuff_ind, j, train_block_size):
                    assert not (np.any(np.isnan(x)))
                    train_ind = shuff_ind - j*train_block_size
                    X_train[train_ind,:] = x
                    Y_train[train_ind] = y
            for index in neg_samples:
                x = X[index:index+padded_window,:]
                y = 0
                assert not (np.any(np.isnan(x)))
                shuff_ind = train_shuffle[train_index]
                train_index +=1
                if in_current_block(shuff_ind, j, train_block_size):
                    train_ind = shuff_ind - j*train_block_size
                    X_train[train_ind,:] = x
                    Y_train[train_ind] = y

        folder = "../training_windows/label_model_windows/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        x_train_name = "X_train_npm_{}_rnm_{}_num_{}".format(
                near_pos_multiple, rand_neg_multiple, j)
        y_train_name = "Y_train_npm_{}_rnm_{}_num_{}".format(
                near_pos_multiple, rand_neg_multiple, j)
        np.save(folder + x_train_name, X_train)
        np.save(folder + y_train_name, Y_train)

        print('Finished Training Window Block {} of size:'.format(j))
        print(train_block_size, X_train.shape, Y_train.shape)
        X_train, Y_train = None, None

        assert(train_index == num_train)

def save_trainval_set(near_pos_multiple,rand_neg_multiple, window, padded_window, 
                 num_files, train_files, val_files, num_train, pathname = ''):

    X_trainval = []
    Y_trainval = []       
     
        
    for i in val_files:
        X = np.load('./numpy_data/inputs/inputs_'+ str(i)+'.npy')
        Y = np.load('./numpy_data/labels/labels_'+ str(i)+'.npy')
        
        pos_samples, neg_samples = get_pos_and_neg_samples(X, Y, 
                padded_window, pos_keep, near_pos_keep, rand_neg_multiple)
        size = len(pos_samples) + len(neg_samples)
        des_size = 100000
        if size > des_size:
            new_pos_size = int((des_size/size)*len(pos_samples))
            pos_samples = list(np.random.choice(pos_samples, size=new_pos_size))
            
            new_neg_size = int((des_size/size)*len(neg_samples))
            neg_samples = list(np.random.choice(neg_samples, size=new_neg_size))

        for index in pos_samples:
            x = X[index:index+padded_window,:]
            y = 1
            X_trainval.append(x)
            Y_trainval.append(y)

        for index in neg_samples:
            x = X[index:index+padded_window,:]
            y = 0
            X_trainval.append(x)
            Y_trainval.append(y)

    X_trainval = np.stack(X_trainval)
    Y_trainval = np.reshape(np.stack(Y_trainval), (len(Y_trainval),1))

    print('Trainval set shapes:')
    print('X_trainval: ', X_trainval.shape)
    print('Y_trainval: ', Y_trainval.shape)

    #save trainval sets
    folder = "../training_windows/label_model_windows/"
    x_trainval_name = "X_trainval_npm_{}_rnm_{}.npy".format(
                near_pos_multiple, rand_neg_multiple)
    y_trainval_name = "Y_trainval_npm_{}_rnm_{}.npy".format(
                near_pos_multiple, rand_neg_multiple)
    np.save(folder + x_trainval_name, X_trainval) 
    np.save(folder + y_trainval_name, Y_trainval)

save_train_set(near_pos_multiple,rand_neg_multiple, window, padded_window, 
                 num_files, train_files, val_files, num_train, pathname = '')

save_trainval_set(near_pos_multiple,rand_neg_multiple, window, padded_window, 
                 num_files, train_files, val_files, num_train, pathname = '')
