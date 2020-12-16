import numpy as np
import math
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import sys
sys.path.insert(0,'../preprocessing')
from data_config import config

# load dataset info
data_config = config()
fs = data_config.fs
num_features = data_config.num_features
correction_window_s = data_config.correction_window_s
max_exp_perturbation = data_config.max_exp_perturbation
moving_average_len = data_config.moving_average_len


def smooth_signal(y_pred):
    """
    Smooths the time series y_pred using a moving average filter.
    We return the time series z given by
    z[n] = (1/m) * (y_pred[n-(m-1)/2] + ... + y_pred[n+(m-1)/2])
    The parameter m is determined by the moving_average_len
    parameter in preprocessing/data_config.
    """

    smooth_y = np.zeros((len(y_pred),1))
    
    low = -math.floor(moving_average_len/2)
    high = low + moving_average_len
    for i in range(len(smooth_y)):
        nearby_vals = [y_pred[i+j] if (0<=i+j and i+j < len(y_pred)) 
                else 0 for j in range(low, high)]
        smooth_y[i] = np.mean(nearby_vals)
    return smooth_y


def consolidate_prediction_times(
    positive_times, y_pred, time_to_pred_index, chaining_dist=5, threshold=0.9
):
    """
    Consolidates the prediction times in positive times. Note
    that the network will normally output more than one prediction
    for a given true lunge, since we predict every second.
    This function takes the network outputs and tries to
    return predictions  which contain exactly one prediction
    corresponding to each true lunge.

    Specifically, two times in positive times are
    joined together if the times are within
    chaining_dist seconds of each other. Ideally
    each collection of times corresponds to a single
    true lunge. For each collection of times, the time
    in the collection with the maximum probability
    prediction is output, provided that the probability
    exceeds the threshold.

    :param positive_times: (list) list of times with network prediction
                                probabilities greater than 0.5
    :param y_pred: (numpy array) model prediction probabilities at each second
    :param time_to_pred_index: (dict) maps a time (in seconds) to the
                                corresponding index in y_pred
    :param chaining_dist: (float) distance used for chaining
    :param threshold: (float) probability used for thresholding
    :return: consolidation_times: list of consolidated prediction times
    """

    partitions = []
    l = []
    if len(positive_times) > 0:
        prev = positive_times[0]
        l.append(positive_times[0])
        
    for i in range(1,len(positive_times)):
        curr = positive_times[i]
        if (curr-prev) < chaining_dist:
            l.append(curr)
        else:
            partitions.append(l)
            l = [curr]
        prev = curr

    consolidated_times = []
    f = lambda x: y_pred[time_to_pred_index[x]]
    for l in partitions:
        m = max(l,key=f)
        if f(m) > threshold:
            consolidated_times.append(m)
    return consolidated_times


def correct_samples_regression(positive_samples, features, correction_model):
    """
    Uses the regression correction model to place the output lunge times
    closer to the true lunge times.
    For more details, see the correct_samples function.
    """
    scaling_factor = max_exp_perturbation
    window_s = correction_window_s

    window = window_s*fs
    centered_windows = [features[s-int(window/2):s+int(window/2),:]
                        for s in positive_samples]
    w, f = centered_windows[0].shape
    centered_windows = [np.reshape(window, (1, w*f)) for window in centered_windows]

    delta_positive_samples = [scaling_factor*fs*correction_model.predict(window)
                              for window in centered_windows]

    corrected_samples = [int(round(delta.item()+s)) for delta, s in
                         zip(delta_positive_samples, positive_samples)]
    return corrected_samples


def delta_class(window, correction_model, max_exp_perturbation):
    """
    Helper function for correct_samples_class.
    """
    probs = correction_model.predict(window)
    pred = np.argmax(probs)

    return pred - fs*max_exp_perturbation


def correct_samples_class(positive_samples, features, correction_model):
    """
    Uses the regression correction model to place the output lunge times
    closer to the true lunge times.
    For more details, see the correct_samples function.
    """
    window_s = correction_window_s

    window = window_s*fs
    centered_windows = [features[s-int(window/2):s+int(window/2), :] for s in positive_samples]
    w, f = centered_windows[0].shape
    centered_windows = [np.reshape(window, (1, w*f)) for window in centered_windows]
    delta_positive_samples = [
        delta_class(window, correction_model, max_exp_perturbation)
        for window in centered_windows
    ]
    corrected_samples = [int(round(delta+s)) for delta, s in
                         zip(delta_positive_samples, positive_samples)]
    return corrected_samples


def correct_samples(positive_samples, features, correction_model, corr_model_type):
    """
    Takes the sample predictions given by positive_samples and uses
    the correction model to place them closer to the true
    lunge times.

    The correction process is as follows:
    For each prediction sample s in positive_samples, a window
    of features centered at s is fed into the correction model.
    The correction model predicts predict the location of the true
    lunge in the centered window. Then the sample s is moved to
    the predicted location of the true lunge.

    :param positive_samples: (list) list of consolidated prediction
                                    samples
    :param features: (np array (T, num_features)) features of input
                                                    deployment
    :param correction_model: keras model used for corrections
    :param corr_model_type: (str) type of correction model. Either
                            'classification, 'regression'
    :return: (list) list of corrected_samples
    """
    if corr_model_type == 'regression':
        return correct_samples_regression(positive_samples, features, correction_model)
    elif corr_model_type == 'classification':
        return correct_samples_class(positive_samples, features, correction_model)
    else:
        raise('Invalid model type ' + str(corr_model_type))


def get_predictions(
        features, model, flattened_input, correction_model=None,
        corr_model_type=None, y_pred=None, chaining_dist=5., threshold=0.9
):
    """
    Returns the lunge prediction times and samples for the
    input deployment features. The predictions are obtained  using the
    given labeling model and correction model.

    :param features: (np array (T, num_features)) features of input
                                                    deployment
    :param model: keras model used for labeling
    :param flattened_input: whether the input to model is flat
    :param correction_model: keras model used for corrections
    :param corr_model_type: type of correction model. Either
                            'classification', 'regression',
                            or None
    :param y_pred: (numpy array) model prediction probabilities at
                                each second. If None, the
                                y_pred is computed
    :param chaining_dist: (float) chaining distance used for corrections
    :param threshold: (float) threshold used for corrections
    :return: (list) prediction samples
             (list) prediction times
    """

    skip = 1
    if flattened_input:
        samples_per_window = int(model.input_shape[1]/num_features)
    else:
        samples_per_window = model.input_shape[1]

    if y_pred is None:
        y_pred = get_y_pred(features, model, flattened_input)
    m=y_pred.shape[0]
    # smooth predictions
    y_pred = smooth_signal(y_pred)

    samples = [k*skip*fs+int(samples_per_window/2) for k in range(0,m)]
    times = [int(s/fs) for s in samples]

    time_to_pred_index = dict([(t, i) for i, t in enumerate(times)])
    time_to_sample = dict([(t, s) for t, s in zip(times, samples)])

    positive_times = [t for t in times if y_pred[time_to_pred_index[t]][0] > 0.5]

    positive_times = consolidate_prediction_times(
        positive_times, y_pred, time_to_pred_index, chaining_dist, threshold
    )
    positive_samples = [time_to_sample[t] for t in positive_times]

    if correction_model is not None:
        positive_samples = correct_samples(
            positive_samples, features, correction_model, corr_model_type
        )

    positive_times = [s/fs for s in positive_samples]

    return positive_samples, positive_times


def get_y_pred(features, model, flattened_input):
    """
    Returns the model predictions on the deployment given
    by features.

    The predictions are computed by inputting
    the windows at times [t,  t + window_size) to the model.
    The value of t sweeps over the deployment in 1 second
    increments.

    :param features: (np array (T, num_features)) input deployment
    :param model: keras model used for labeling
    :param flattened_input: whether model inputs a flattened window
    :return: y_pred: numpy array of model prediction probabilities
                    at each second
    """
    # load model
    if flattened_input:
        samples_per_window = int(model.input_shape[1]/num_features)
    else:
        samples_per_window = model.input_shape[1]    
    secs_per_window = int(samples_per_window/fs)
    total_seconds = int(features.shape[0]/fs)
    
    # make predictions
    end = math.floor(total_seconds)-secs_per_window
    skip = 1 # predict every skip seconds
    m = math.ceil(end/skip)
    
    flattened_dim = num_features*samples_per_window
    X = np.zeros((m,samples_per_window,num_features))
    
    for j in range(0, m):
        sec = j*skip
        sample = int(sec*fs) # (make sure is already integer)
        window = features[sample:sample+samples_per_window,:]
        X[j, :, :] = window

    if flattened_input:
        X = np.reshape(X, (X.shape[0], flattened_dim))

    y_pred = model.predict(X)
    return y_pred


def get_model_metrics(evaluation_files, model, flattened_input, tolerance_s,
                      correction_model=None, corr_model_type=None,
                      chaining_dists = [5], thresholds = [0.9]):
    """
    Computes the evaluation metrics for on the deployments specified
    in evaluation files. These metrics are discussed in the
    Readme file. In addition, the chaining distance and
    threshold that optimize the f_1 score are computed.

    The results are returned in a dictionary best_model_metrics.

    :param evaluation_files: (list of ints) list of file numbers
                            used for evaluation
    :param model: keras model used for predictions
    :param flattened_input: (bool) whether model inputs flattened
                                windows
    :param tolerance_s: (int) the tolerance (in seconds) for a
                            prediction to be considered correct
    :param correction_model: keras model used for prediction correction
    :param corr_model_type: type of correction model. Either
                            'classification', 'regression',
                            or None
    :param chaining_dists: (list of floats) list of chaining distances
                                to use in evaluation
    :param thresholds: (list of floats) list of thresholds
                                to use in evaluation
    :return: (dict) best_model_metrics containing evaluation statistics
                    and parameters which optimize the f_1 score on
                    evaluation_files
    """

    y_preds = []
    for file_num in evaluation_files:
        features= np.load('../preprocessing/numpy_data/inputs/inputs_'+ str(file_num)+'.npy')
        labels = np.load('../preprocessing/numpy_data/labels/labels_'+ str(file_num)+'.npy')
        y_preds.append(get_y_pred(features, model, flattened_input))

    best_model_metrics = {}
    best_f_1 = -float('inf')
    chaining_dists = sorted(chaining_dists, reverse=True)
    thresholds = sorted(thresholds, reverse=True)
    for chaining_dist, threshold in itertools.product(chaining_dists, thresholds):
        tot_correct_dist = 0  # dist in seconds
        total_correct = 0
        total_true = 0
        total_pred = 0
        total_overcounted = 0
        distances = []
        for y_pred, file_num in zip(y_preds, evaluation_files):
            # load
            feat_prefix = '../preprocessing/numpy_data/inputs/inputs_'
            lab_prefix = '../preprocessing/numpy_data/labels/labels_'
            features = np.load(feat_prefix + str(file_num)+'.npy')
            labels = np.load(lab_prefix + str(file_num)+'.npy')

            # get predictions
            positive_samples, __ = get_predictions(
                features, model, flattened_input, correction_model,
                corr_model_type, y_pred=y_pred, chaining_dist=chaining_dist,
                threshold=threshold)
            
            tolerance = tolerance_s*fs  # tolerance window for correct prediction
            sum_dist = 0

            true_lunge_samples = np.where(labels == 1)[0]
            num_correct = 0
            num_true = len(true_lunge_samples)
            if not positive_samples:
                print('No positive predictions for this deployment')
            else:
                for true_lunge_sample in true_lunge_samples:
                    # TODO: We really should be making an assignment from our predictions to
                    # to the labels; otherwise, one prediction could count for multiple true labels
                    dist = min(abs(true_lunge_sample - s) for s in positive_samples)
                    if dist < tolerance:
                        sum_dist += dist/fs
                        distances.append(dist/fs)
                        num_correct += 1
            
            tot_correct_dist += sum_dist
            total_correct += num_correct
            total_true +=num_true
            
            already_predicted = set([])
            num_pred = len(positive_samples)
            total_pred += num_pred

            # calculate num overcounted
            num_overcounted = 0
            if len(true_lunge_samples) == 0:
                num_overcounted = len(positive_samples)
                print('No positive labels for this deployment')
            else:
                for sample in positive_samples:
                    dist, true_lunge_sample = min((abs(sample - s), s)
                                                 for s in true_lunge_samples)
                    if dist < tolerance:
                        if true_lunge_sample in already_predicted:
                            num_overcounted += 1
                        already_predicted.add(true_lunge_sample)

            total_overcounted += num_overcounted
        
        if total_pred == 0:
            continue
        
        tp = round(total_correct/total_true, 3)
        fp = round(1-total_correct/total_pred, 3)
        
        precision = total_correct/total_pred
        recall = total_correct/total_true
        beta = 2
        f_1 = round(2 * (precision*recall)/(precision+recall), 3)
        f_2 = round((1 + beta**2)*(precision*recall)/(beta**2*precision+recall), 3)
        if f_1 > best_f_1:
            best_model_metrics = {
                'tp': tp,
                'fp': fp,
                'total_true': total_true,
                'total_correct': total_correct,
                'total_pred': total_pred,
                'total_overcounted': total_overcounted,
                'avg_error': (tot_correct_dist/total_correct),
                'distances': distances,
                'tolerance_s': tolerance_s,
                'f_1': f_1,
                'f_2': f_2,
                'chaining_dist': chaining_dist,
                'threshold': threshold
            }
            best_f_1 = f_1
        
    return best_model_metrics


def print_model_metrics(model_metrics):
    """
    Prints the evaluation metrics given in model_metrics.
    """
    tolerance_s = model_metrics['tolerance_s']
    print(
        'The true positive rate for tolerance ', tolerance_s, ' seconds is ',
        model_metrics['tp']
    )
    print(
        'The false positive rate for tolerance ',  tolerance_s, ' seconds is ',
        model_metrics['fp']
    )
    
    print('Total lunges in files: ', model_metrics['total_true'])
    print('Num correct lunges: ', model_metrics['total_correct'])
    print('Num predicted lunges: ', model_metrics['total_pred'])
    print('We overcount by ', model_metrics['total_overcounted'])
    print(
        'We are off by an average of this many seconds: ',
        model_metrics['avg_error']
    )
    print('The f_1 score is ', model_metrics['f_1'])
    print('The f_2 score is ', model_metrics['f_2'])
    
    distances = model_metrics['distances']
    plt.hist(distances,bins=[i*.25 for i in range(41)])
    plt.xlabel('Prediction Error (seconds)')
    plt.ylabel('Number of Predictions')
    plt.title('Error Histogram For Label Predictions')
