import random
import numpy as np 
from keras import backend as K
import tensorflow as tf

class feed_forward_config():
    def __init__(self):

        # model and training config
        self.flattened_input = True #True for feed_forward
        self.near_pos_multiple = 0.2
        self.rand_neg_multiple = 2.8
        self.num_train_sets = 10
            #these 3 lines determine the training set in case multiple
            #training sets have been generated.
        self.hyper_search = False
            #whether to do random hyperparameter search
        self.model_name = 'feed_forward'

        # hyper params
        self.chaining_dists = [i for i in range(3,11)]
        self.thresholds = np.linspace(0.5, 0.9, 10)
            #used for consolidation of training labels
        self.tolerance_seconds = 5
            #model predicted label is considered correct if it is at most 
            #tolerance_seconds away from a ground truth label
        self.n_iterations = 20
            #number of training epochs
        self.batch_size = 128
        self.optimizer = 'adam'
        self.learning_rate = 1e-3
        self.hidden_layers = [32, 20]
            #number of nodes in each hidden layer 
        self.l2_reg = 0.0 
            #l2 regularization parameter
        self.batch_norm = True 
            #whether to use batch normalization
        self.activation = 'relu'
            #which nonlinear activation to use
        self.output_activation = 'sigmoid'
        self.pos_weight = 2 #*
            #weight of positive class for weighted binary cross entropy
            #(pos_weight=1 corresponds to normal binary cross entropy)
        self.loss = get_weighted_bce(self.pos_weight)
        self.metrics=['accuracy']

class resnet_config():
    def __init__(self):
        #See ff_config comments for most variable descriptions

        # model and training config
        self.flattened_input = False #False for resnet
        self.near_pos_multiple = 0.2
        self.rand_neg_multiple = 2.8
        self.num_train_sets = 10 
        self.hyper_search = False 
        self.model_name = 'resnet'

        # hyper params
        self.chaining_dists = [i for i in range(3,11)] 
        self.thresholds = np.linspace(0.5, 0.9, 4)
        self.tolerance_seconds = 5 
        self.n_iterations = 20
        self.batch_size = 128 
        self.optimizer = 'adam'
        self.learning_rate = 1e-3
        self.pos_weight = 2 
        self.loss = get_weighted_bce(self.pos_weight)
        self.metrics=['accuracy']

class feed_forward_search_config():
    def __init__(self):

        # model and training config
        self.flattened_input = True
        self.near_pos_multiple = 0.2
        self.rand_neg_multiple = 2.8 
        self.num_train_sets = 10 
        self.hyper_search = True
        self.model_name = 'feed_forward'

        # hyper params
        self.chaining_dists = [i for i in range(3,11)]
        self.thresholds = np.linspace(0.5, 0.9, 10)
        self.tolerance_seconds = 5
        self.n_iterations = 1
        self.batch_size = 128
        self.optimizer = 'adam'
        exp = -(2*np.random.random()+2)
        self.learning_rate = 10**exp
        h_layers = random.sample([[32, 20], [64, 32, 20], [128, 64, 32, 20]], 1)[0]
        self.hidden_layers = [64, 32, 20]
        reg = random.sample([0., 1e-4, 1e-2, 1.], 1)[0]
        self.l2_reg = reg
        self.batch_norm = True
        self.activation = 'relu'
        self.output_activation = 'sigmoid'
        # self.loss = 'binary_crossentropy'
        #weight = random.sample([], 1)[0]
        exp = 0.26 + 0.74*np.random.random()
        self.pos_weight = 2**exp
        self.loss = get_weighted_bce(self.pos_weight)
        self.metrics=['accuracy']

class cnn_rnn_search_config():
    def __init__(self):
        #See ff_config comments for most variable descriptions

        # model and training config
        self.flattened_input = False #False for resnet
        self.near_pos_multiple = 0.2
        self.rand_neg_multiple = 2.8
        self.num_train_sets = 10 
        self.hyper_search = True 
        self.model_name = 'resnet'

        # hyper params
        self.chaining_dists = [i for i in range(3,11)] 
        self.thresholds = np.linspace(0.5, 0.9, 10)
        self.tolerance_seconds = 5 
        self.n_iterations = 1
        self.batch_size = 128 
        self.optimizer = 'adam'
        exp = -(0.6*np.random.random()+2.7)
        self.learning_rate = 10**exp
        exp = 0.26 + 0.74*np.random.random()
        self.pos_weight = 2**exp
        self.loss = get_weighted_bce(self.pos_weight)
        self.metrics=['accuracy']

def get_config(name):
    if name =='feed_forward':
        return feed_forward_config()
    elif name =='resnet':
        return resnet_config()
    elif name == 'feed_forward_search':
        return feed_forward_search_config()
    elif name == 'cnn_rnn_search':
        return cnn_rnn_search_config()

def get_weighted_bce(pos_weight):
    def weighted_bce(y_true,y_pred):
        epsilon = 1e-7
        output = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        logits = tf.log(output / (1 - output))
        return (1/(1+pos_weight))*tf.nn.weighted_cross_entropy_with_logits(
                targets=y_true,
                logits=logits,
                pos_weight=pos_weight,
            )
    return weighted_bce

def asym_binary_focal_loss(gamma0=1., gamma1=0., alpha=.25):
    def asym_binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return K.mean(-alpha * K.pow(1. - pt_1, gamma1) * K.log(pt_1) \
                -(1 - alpha) * K.pow(pt_0, gamma0) * K.log(1. - pt_0))
    return asym_binary_focal_loss_fixed

