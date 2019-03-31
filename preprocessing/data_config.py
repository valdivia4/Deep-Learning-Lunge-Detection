import numpy as np 

class config():

    num_train_sets = 10 
    species_code = 'bb' 
    window_s = 4

    ## near_pos_multiple is the multiple of near positive training examples to keep relative to positive training examples
    ## random_neg_multiple is the multiple of randomly selected negative examples

    near_pos_multiple = 0.2 ##make sure fs*multiple is an integer, usually fs=10
    rand_neg_multiple = 19.8
       
    num_features = 5
    SAMPLES_PER_S = 10
    padded_window_s = 20

    num_files = 6
    train_files = [0,1,2,4]
    val_files = [5]
    test_files = [3]