#!/usr/bin/env python
# coding: utf-8


from keras import backend as K
import tensorflow as tf
import datetime
import sys
sys.path.insert(0,'../utils')
from configs import get_config
from train import run_training

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config = tf_config)
K.tensorflow_backend._get_available_gpus()

config = get_config('resnet')

def get_weighted_bce(pos_weight):
    def weighted_bce(y_true,y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(
                y_true,
                y_pred,
                pos_weight,
            )
    
    return weighted_bce


# From Github repo: https://github.com/hfawaz/dl-4-tsc


# ResNet
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import keras 
import numpy as np 
import pandas as pd 
import time

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 
from resnet_utils.utils import save_logs

class Classifier_RESNET: 

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init.hdf5')

	def build_model(self, input_shape, nb_classes):
		n_feature_maps = 64

		input_layer = keras.layers.Input(input_shape)
		
		# BLOCK 1 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_1 = keras.layers.add([shortcut_y, conv_z])
		output_block_1 = keras.layers.Activation('relu')(output_block_1)

		# BLOCK 2 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_2 = keras.layers.add([shortcut_y, conv_z])
		output_block_2 = keras.layers.Activation('relu')(output_block_2)

		# BLOCK 3 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# no need to expand channels because they are equal 
		shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

		output_block_3 = keras.layers.add([shortcut_y, conv_z])
		output_block_3 = keras.layers.Activation('relu')(output_block_3)

		# FINAL 
		
		gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

		output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5' 

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model
	
	def fit(self, x_train, y_train, x_val, y_val): 
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 32
		nb_epochs = 1

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

		duration = time.time() - start_time

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

# 		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()


# From Github repo: https://github.com/hfawaz/dl-4-tsc

from resnet_utils.utils import create_directory


output_directory = 'results/'

path_to_trainset = '../data/label_model_windows/'
X_trainval = np.load(path_to_trainset + 'X_trainval_npm_{}_rnm_{}.npy'.format(config.near_pos_multiple, config.rand_neg_multiple)) 

#output_directory = create_directory(output_directory)
input_shape = X_trainval.shape[1:]
nb_classes = 2

resnet = Classifier_RESNET(output_directory, input_shape, nb_classes, verbose=True)
model = resnet.build_model(input_shape, nb_classes)

run_training(model, config)

# if species_code == 'bw':
#     val = [10, 6, 9]
#     test = [3, 20, 17]
#     train= [i for i in range(29) if i not in val and i not in test]
# else:
#     val = [5]
#     test = [3]
#     train = [i for i in range(6) if i not in val and i not in test]
# evaluation_files = val


# tolerance_seconds = 5





# n_iterations = 20
# epochs = 0

# model_type = 'Resnet'
# experiment_time = datetime.datetime.now()


# for i in range(n_iterations):
#     for j in range(num_train_sets):
#         X_train = np.load(path_to_trainset + 'X_train_npm_{}_rnm_{}_num_{}.npy'.format(near_pos_multiple, rand_neg_multiple,j))
#         Y_train = np.load(path_to_trainset +  'Y_train_npm_{}_rnm_{}_num_{}.npy'.format(near_pos_multiple, rand_neg_multiple,j))
        
#         model.fit(X_train, Y_train, batch_size=128, epochs=1, verbose=1, callbacks=None, validation_data=(X_trainval, Y_trainval))
#         X_train, Y_train = None,None
#     epochs += 1
#     # tp, fp, f_1, f_2 = get_tp_fp_f1_f2(evaluation_files, model, model_type, species_code, tolerance_seconds)
#     model_metrics = get_model_metrics(evaluation_files, model, model_type, species_code, tolerance_seconds)
#     tp, fp, f_1, f_2 = model_metrics['tp'], model_metrics['fp'], model_metrics['f_1'], model_metrics['f_2']
#     model.save('../models/label_models/Resnet_species_code_{}_e_{}_tp_{}_fp_{}_f_1_{}_f_2_{}'.format(species_code,epochs,tp,fp,f_1,f_2))

    




# print (epochs)






