import tensorflow as tf
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import datetime
import sys
sys.path.insert(0,'../utils')
from utils import get_model_metrics
sys.path.insert(0,'../preprocessing')
from data_config import config as d_config

def run_training(model, config):
	data_config = d_config()
	train = data_config.train_files
	val = data_config.val_files
	test = data_config.test_files

	evaluation_files = val


	tolerance_seconds = config.tolerance_seconds
	ep = 0

	#TODO integrate model type with flattened_input
	model_name =  config.model_name
	experiment_time = datetime.datetime.now()
	time_as_string = experiment_time.ctime().replace(' ', '_').replace(':', '-')
	output_directory = '../models/label_models/' + '{}_{}'.format(model_name, time_as_string) 
	os.makedirs(output_directory)
	#save config
	with open(output_directory+'/config.txt', 'w') as f:
		f.write(''.join(["%s = %s\n" % (k,v) for k,v in config.__dict__.items()]))

	path_to_trainset = '../training_windows/label_model_windows/'
	X_trainval = np.load(path_to_trainset + 'X_trainval_npm_{}_rnm_{}.npy'.format(config.near_pos_multiple, config.rand_neg_multiple))
	Y_trainval =  np.load(path_to_trainset + 'Y_trainval_npm_{}_rnm_{}.npy'.format(config.near_pos_multiple, config.rand_neg_multiple))
	if config.flattened_input: 
		m_trainval, w, f = X_trainval.shape
		X_trainval = np.reshape(X_trainval, (m_trainval, w*f))

	for i in range(config.n_iterations):
	    for j in range(config.num_train_sets):
	        X_train = np.load(path_to_trainset + 'X_train_npm_{}_rnm_{}_num_{}.npy'.format(config.near_pos_multiple, config.rand_neg_multiple,j))
	        Y_train = np.load(path_to_trainset +  'Y_train_npm_{}_rnm_{}_num_{}.npy'.format(config.near_pos_multiple, config.rand_neg_multiple,j))
	        
	        if config.flattened_input:
	        	m_train, w, f = X_train.shape
	        	X_train = np.reshape(X_train, (m_train, w*f))

	        model.fit(X_train, Y_train, batch_size=config.batch_size, epochs=1, verbose=1, callbacks=None, validation_data=(X_trainval, Y_trainval))
	        X_train, Y_train = None, None
	    ep += 1
	    model_metrics = get_model_metrics(evaluation_files, model, config.flattened_input, tolerance_seconds, chaining_dists = config.chaining_dists, thresholds = config.thresholds)
	    tp, fp, f_1, f_2, chain, thresh = model_metrics['tp'], model_metrics['fp'], model_metrics['f_1'], model_metrics['f_2'], model_metrics['chaining_dist'], model_metrics['threshold']
	    print (tp, fp, f_1, f_2, chain, thresh)
	    model.save(output_directory + '/ep_{}_tp_{}_fp_{}_f_1_{}_f_2_{}_chain_{}_thresh_{}'.format(ep,tp,fp,f_1,f_2, chain, thresh))
