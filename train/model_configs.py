import numpy as np 

class resnet_config():

    # model and training config
    flattened_input = False
    near_pos_multiple = 0.2
    rand_neg_multiple = 19.8 
    num_train_sets = 10
    model_name = 'resnet'

    # hyper params
    chaining_dists = [i for i in range(1,11)]
    thresholds = np.linspace(0.5, 0.9, 10)
    tolerance_seconds = 5
    n_iterations = 20
    batch_size = 128
    optimizer = 'adam'
    learning_rate = 1e-3
    hidden_layers = [32,20]
    batch_norm = True
    activation = 'relu'
    output_activation = 'sigmoid'
    loss = 'binary_crossentropy'
    metrics=['accuracy']

class ff_config():

    # model and training config
    flattened_input = True
    near_pos_multiple = 0.2
    rand_neg_multiple = 19.8 
    num_train_sets = 10
    model_name = 'feed_forward'

    # hyper params
    chaining_dists = [i for i in range(1,11)]
    thresholds = np.linspace(0.5, 0.9, 10)
    tolerance_seconds = 5
    n_iterations = 20
    batch_size = 128
    optimizer = 'adam'
    learning_rate = 1e-3
    hidden_layers = [32,20]
    batch_norm = True
    activation = 'relu'
    output_activation = 'sigmoid'
    loss = 'binary_crossentropy'
    metrics=['accuracy']

def get_config(name):
    if name =='feed_forward':
        return ff_config()
    elif name =='resnet':
        return resnet_config()
