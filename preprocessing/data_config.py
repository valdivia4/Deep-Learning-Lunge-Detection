import numpy as np 

class config():

    def __init__(self):

        self.num_train_sets = 10 
        self.window_s = 4
        self.padded_window_s = 20

        ## near_pos_multiple is the multiple of near positive training examples to keep relative to positive training examples
        ## random_neg_multiple is the multiple of randomly selected negative examples

        self.near_pos_multiple = 0.2 ##make sure fs*multiple is an integer, usually fs=10
        self.rand_neg_multiple = 19.8
           
        self.num_features = 5
        self.fs = 10

        # Correction windows
        self.num_correction_windows_per_label = 10
        self.correction_window_s = 16
        self.max_exp_perturbation = 5

        self.num_files = 6
        self.train_files = [0,1,2,4]
        self.val_files = [5]
        self.test_files = [3]
