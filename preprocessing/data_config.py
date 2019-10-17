import numpy as np 

class config():
    """Class containing variables used to generate training sets."""
    def __init__(self):

        # SET THE VARIABLES BELOW ACCORDING TO YOUR DATA SPECIFICATIONS:

        self.num_train_sets = 10 
        self.window_s = 4 
            # number of seconds for inner window, e.g.
            # window_s = 4 means a window is labeled 1 if
            # there is a lunge within 2 seconds of the middle
        
        self.padded_window_s = 20 
            # total window seconds

        self.moving_average_len = 3
            # final probability at time t is the average of the neural network
            # probabilities at times t-floor(moving_average_len/2)...
            # t + floor(moving_average_len/2) (in seconds)
            # a good setting should be roughly equal to window_s
            # value should be odd

        self.near_pos_multiple = 0.2 
        self.rand_neg_multiple = 19.8
            # near_pos_multiple is the multiple of near positive training
            #  examples to keep relative to total positive training examples
            # (lunge is in the padded_window but not in the window)
            # random_neg_multiple is the multiple of randomly selected
            # negative examples
           
        # Correction model settings
        self.num_correction_windows_per_label = 10 
            # number of windows per lunge
        self.correction_window_s = 16 
            # number of seconds in correction window
        self.max_exp_perturbation = 5 
            # maximum expected mistake in labeling in seconds
        self.std = 1.
            # standard deviation of normal distribution used to generate
            # perturbed window

        # training set info
        self.num_features = 5
        self.fs = 10

        self.num_files = 6
        self.train_files = [0, 1, 2, 4]
        self.val_files = [5]
        self.test_files = [3]
