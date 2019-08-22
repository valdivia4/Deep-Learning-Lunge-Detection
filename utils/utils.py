import numpy as np
import keras
import keras.backend as K
import math
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import sys
sys.path.insert(0,'../preprocessing')
from data_config import config

data_config = config()
fs = data_config.fs
num_features = data_config.num_features
correction_window_s = data_config.correction_window_s
max_exp_perturbation = data_config.max_exp_perturbation


def convert_unlabeled_deployment(inputs_file):
    ## converts unlabeled deployment from matlab to python

    inputs_dict = sio.loadmat(inputs_file)
    p = inputs_dict['p']
    roll = inputs_dict['roll']
    pitch = inputs_dict['pitch']
    speed = inputs_dict['speed_JJ']
    jerk = inputs_dict['jerk']
    n = speed.shape[0]
    #normalize
    temp = [p, roll, pitch, jerk, speed]
    for i,v in enumerate(temp):
        ave = np.nanmean(v)
        std = np.nanstd(v)
        v = (v-ave)/std
        temp[i] = v

    features = []
    for feature in temp:
        df = pd.DataFrame(feature)
        df = df.fillna(method='ffill').fillna(method='bfill')
        assert not (np.any(np.isnan(df)))
        features.append(df.values)
    features = np.squeeze(features).T
    return features

def smooth_signal(y_pred, weights):
    ##Smooths a signal using a list of weights (weights should be length 5)
    smooth_y = np.zeros((len(y_pred),1))

    for i in range(len(smooth_y)):
        nearby_vals = [y_pred[i+j]*weights[2+j] if (0<=i+j and i+j < len(y_pred)) else 0 for j in range(-2,3) ]
        smooth_y[i] = np.mean(nearby_vals)
    return smooth_y

def consolidate_times(positive_times, y_pred, time_to_pred_index, chaining_dist = 5, threshold = 0.9):
    #consolidate nearby predictions to output one label per lunge
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

def correct_samples(positive_samples, features, correction_model, fs):
    ## Places the output lunge times closer to the true lunge times
    scaling_factor = max_exp_perturbation
    window_s = correction_window_s

    WINDOW = window_s*fs 
    centered_windows = [features[s-int(WINDOW/2):s+int(WINDOW/2),:] for s in positive_samples]
    w,f = centered_windows[0].shape
    centered_windows = [np.reshape(window,(1,w*f)) for window in centered_windows]
    delta_positive_samples = [scaling_factor*fs*correction_model.predict(window) for window in centered_windows]
    corrected_samples = [int(round(delta.item()+s)) for delta,s in zip(delta_positive_samples, positive_samples)]
    return corrected_samples

def get_predictions(features, model, flattened_input, correction_model=None, y_pred=None, chaining_dist=5, threshold=0.9):
    skip = 1
    if flattened_input:
        samples_per_window = int(model.input_shape[1]/num_features)
    else:
        samples_per_window = model.input_shape[1]

    if y_pred is None:
        y_pred = get_y_pred(features, model, flattened_input)
    m=y_pred.shape[0]
    ##smooth predictions
    weights = [1,1,1,1,1]
    y_pred = smooth_signal(y_pred,weights)

#     times = np.linspace(int(secs_per_window/2), m*skip + int(secs_per_window/2),m)
    samples = [k*skip*fs+int(samples_per_window/2) for k in range(0,m)]
    times = [int(s/fs) for s in samples]

    time_to_pred_index = dict([(t,i) for i,t in enumerate(times)])
    time_to_sample = dict([(t,s) for t,s in zip(times,samples)])

    positive_times = [t for t in times if y_pred[time_to_pred_index[t]][0] > 0.5]

    positive_times = consolidate_times(positive_times, y_pred, time_to_pred_index, chaining_dist, threshold)
    positive_samples = [time_to_sample[t] for t in positive_times]

    if correction_model is not None:
        positive_samples = correct_samples(positive_samples, features,correction_model, fs)

    positive_times = [s/fs for s in positive_samples]

    return positive_samples, positive_times

def get_y_pred(features, model, flattened_input):
    ##load model
    if flattened_input:
        samples_per_window = int(model.input_shape[1]/num_features)
    else:
        samples_per_window = model.input_shape[1]    
    secs_per_window = int(samples_per_window/fs)
    total_seconds = int(features.shape[0]/fs)
    
    ##make predictions
    end =  math.floor(total_seconds)-secs_per_window
    skip = 1 #predict every skip seconds
    m = math.ceil(end/skip)
    
    # TODO: Handle memory overflow of X!!!!!
    # TODO: Handle memory overflow of X!!!!!
    if flattened_input:
        flattened_dim = num_features*samples_per_window
        X = np.zeros((m,flattened_dim))
    else:
        X = np.zeros((m,samples_per_window,num_features))
    
    for j in range(0, m):
        sec = j*skip
        sample = int(sec*fs) #(make sure is already integer)
        window = features[sample:sample+samples_per_window,:]
        if flattened_input:
            window = np.reshape(window, (1,flattened_dim))
            X[j,:] = window
        else:
            X[j,:,:] = window

    y_pred = model.predict(X)
    return y_pred


# def get_tp_fp_f1_f2(evaluation_files, model, model_name, species_code, tolerance_s, correction_model = None):
    
#     y_preds = []
#     for file_num in evaluation_files:
#         features = np.load('../preprocessing/converted_python_data/' + species_code + '/Inputs_' + str(file_num) + '.npy')
#         labels = np.load('../preprocessing/converted_python_data/' + species_code + '/Labels_' + str(file_num) + '.npy')
#         y_preds.append(get_y_pred(features, model, model_name))

#     total_correct = 0
#     total_true = 0
#     total_pred = 0

#     chaining_dist = 5
#     threshold = 0.6
#     for y_pred in y_preds:
#         features = np.load('../preprocessing/converted_python_data/' + species_code + '/Inputs_' + str(file_num) + '.npy')
#         labels = np.load('../preprocessing/converted_python_data/' + species_code + '/Labels_' + str(file_num) + '.npy')
#         if y_pred is None:
#             print('lol')
#         positive_samples, __ = get_predictions(features, model, model_name, correction_model, y_pred=y_pred, chaining_dist=chaining_dist, threshold=threshold)
        
# #         if len(positive_samples) == 0:
# #             positive_samples = [100]
# #             print ('No lunges detected. Guessing a lunge at 10 seconds to continue running.')
        
#         TOLERANCE = tolerance_s*fs #tolerance window for correct prediction
        
#         true_lunge_samples = np.where(labels == 1)[0]
#         num_correct = 0
#         num_true = len(true_lunge_samples)
        
#         # TO-DO: For minke whales, this might be problematic
#         # That is, one prediction could be counted as two if the two lunges are close enough 
#         assigned_true_lunges = set()
#         for predicted_sample in positive_samples:
#             candidates = [s for s in true_lunge_samples if abs(predicted_sample - s) < TOLERANCE and s not in assigned_true_lunges]
#             assignment = min(candidates)
#             assigned_true_lunges.add(assignment)
#         num_correct += len(assigned_true_lunges)

#         # for true_lunge_sample in true_lunge_samples:
#         #     dist = min(abs(true_lunge_sample - s) for s in positive_samples)
#         #     if dist < TOLERANCE:
#         #         num_correct += 1
        
#         total_correct += num_correct
#         total_true += num_true
        
#         num_pred = len(positive_samples)
#         total_pred += num_pred       
        
#     precision = round(total_correct/total_pred, 4)
#     recall = round(total_correct/total_true, 4)
#     beta = 2
#     f_1 = 2 * (precision*recall)/(precision+recall)
#     f_2 = (1 + beta**2)*(precision*recall)/(beta**2*precision+recall)
#     tp = round(total_correct/total_true,3)
#     fp = round(1-total_correct/total_pred, 3)

#     return tp, fp, f_1, f_2

def get_model_metrics(evaluation_files, model, flattened_input, tolerance_s, correction_model = None, chaining_dists = [5], thresholds = [0.9]):

    y_preds = []
    for file_num in evaluation_files:
        features= np.load('../preprocessing/numpy_data/inputs/inputs_'+ str(file_num)+'.npy')
        labels = np.load('../preprocessing/numpy_data/labels/labels_'+ str(file_num)+'.npy')
        y_preds.append(get_y_pred(features, model, flattened_input))

    best_model_metrics = {}
    best_f_1 = -float('inf')
    for chaining_dist, threshold in itertools.product(chaining_dists, thresholds):
        tot_correct_dist = 0 # dist in seconds
        total_correct = 0
        total_true = 0
        total_pred = 0
        total_overcounted = 0
        distances = []
        for y_pred, file_num in zip(y_preds, evaluation_files):
            features= np.load('../preprocessing/numpy_data/inputs/inputs_'+ str(file_num)+'.npy')
            labels = np.load('../preprocessing/numpy_data/labels/labels_'+ str(file_num)+'.npy')
            positive_samples, __ = get_predictions(features, model, flattened_input, correction_model, y_pred=y_pred, chaining_dist=chaining_dist, threshold=threshold)
            
            TOLERANCE = tolerance_s*fs #tolerance window for correct prediction 
            sum_dist = 0

            true_lunge_samples = np.where(labels == 1)[0]
            num_correct = 0
            num_true = len(true_lunge_samples)
            for true_lunge_sample in true_lunge_samples:

                #### TO-DO: We really should be making an assignment from our predictions to
                #### to the labels; otherwise, one prediction could count for multiple true labels
                dist = min(abs(true_lunge_sample - s) for s in positive_samples)
                if dist < TOLERANCE:
                    sum_dist += dist/fs
                    distances.append(dist/fs)
                    num_correct+=1
            
            tot_correct_dist += sum_dist
            total_correct += num_correct
            total_true +=num_true
            
            already_predicted = set([])
            num_pred = len(positive_samples)
            total_pred += num_pred

            ##calculate num overcounted
            num_overcounted = 0
            for sample in positive_samples:
                dist,true_lunge_sample = min((abs(sample - s),s) for s in true_lunge_samples)
                if dist < TOLERANCE:            
                    if true_lunge_sample in already_predicted:
                        num_overcounted += 1
                    already_predicted.add(true_lunge_sample)

            total_overcounted += num_overcounted
            
        tp = round(total_correct/total_true,3)
        fp = round(1-total_correct/total_pred, 3)
        
        precision = total_correct/total_pred
        recall = total_correct/total_true
        beta = 2
        f_1 = round(2 * (precision*recall)/(precision+recall), 3)
        f_2 = round((1 + beta**2)*(precision*recall)/(beta**2*precision+recall), 3)
        if f_1 > best_f_1:
            best_model_metrics = {'tp': tp, 'fp': fp, 'total_true': total_true, 'total_correct': total_correct, 'total_pred': total_pred, 'total_overcounted': total_overcounted, 'avg_error': (tot_correct_dist/total_correct), 'distances': distances, 'tolerance_s': tolerance_s, 'f_1': f_1, 'f_2': f_2, 'chaining_dist' : chaining_dist, 'threshold' : threshold}
            best_f_1 = f_1
        
    return best_model_metrics
    
def print_model_metrics(model_metrics):
    tolerance_s = model_metrics['tolerance_s']
    print ('The true positive rate for tolerance ', tolerance_s,' seconds is ', model_metrics['tp'])            
    print ('The false positive rate for tolerance ', tolerance_s,' seconds is ', model_metrics['fp'])
    
    print ('Total lunges in files: ', model_metrics['total_true'])
    print ('Num correct lunges: ', model_metrics['total_correct'])
    print ('Num predicted lunges: ', model_metrics['total_pred'])
    print ('We overcount by ', model_metrics['total_overcounted'])
    print ('We are off by an average of this many seconds: ', model_metrics['avg_error'])
    print ('The f_1 score is ', model_metrics['f_1'])
    print ('The f_2 score is ', model_metrics['f_2'])
    
    distances = model_metrics['distances']
    plt.hist(distances,bins=[i*.25 for i in range(41)])
    plt.xlabel('Prediction Error (seconds)')
    plt.ylabel('Number of Predictions')
    plt.title('Error Histogram For Label Predictions')
