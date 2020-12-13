import os

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, freqz

from data_config import config 

data_config = config()
use_lowpass_filter = data_config.use_lowpass_filter
order = data_config.order
cutoff = data_config.cutoff
fs = data_config.fs

def butter_lowpass():
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(x):
    b, a = butter_lowpass()
    y = filtfilt(b, a, x, axis=0)
    return y

def filter_data(x):
    # Filter requirements.
    x = butter_lowpass_filter(x)
    return x

def shift(features):
    """
    Normalizes the deployment given by features (so that each
    feature has 0 mean 1 standard deviation)
    """

    ave = np.nanmean(features, axis=0)
    features = features - ave
    return features

def rescale(features):
    num_features = features.shape[1]
    den = np.zeros((1, num_features))
    for f in range(num_features):
        feature = np.abs(features[:, f])
        #feature = feature[~np.isnan(feature)]
        num_features = feature.shape[0]
        ind = feature.argsort()[int(0.99*num_features)]
        den[0, f] = feature[ind]
    return features / den

def clean_data(features):
    """Normalizes the features and removes NaNs"""

    features = shift(features)
    df = pd.DataFrame(features)
    df = df.fillna(axis=0, method='ffill').fillna(axis=0, method='bfill')
    assert not (np.any(np.isnan(df)))
    features = np.array(df)
    if use_lowpass_filter:
        features = filter_data(features)
    features = rescale(features)
    return features


def process_csv(raw_path, filename, is_input=True):
    """
    Reads a multivariate time series from a csv and returns the
    cleaned result as a numpy array.

    :param raw_path: (str) path to the csv
    :param filename: (str) csv file name
    :param is_input: (bool) whether the csv is an input or a label
    :return:
    """

    result = pd.read_csv(raw_path + filename, header=None)
    result = np.array(result)
    if is_input:
        result = clean_data(result)
    return result

def convert_csvs_to_numpy():
    """
    Cleans the csv deployments in the raw_data folder and converts
    them to numpy arrays. The numpy arrays are saved in a newly created
    numpy_data folder.
    """

    for directory in os.listdir("./raw_data"):
        if directory.startswith("."): continue
        raw_path = "./raw_data/" + directory + "/"
        numpy_path = "./numpy_data/" + directory + "/"
        if not os.path.exists(numpy_path):
            os.makedirs(numpy_path)
        for filename in os.listdir(raw_path):
            if filename.endswith('.txt'):
                is_input = filename.startswith('inputs')
                result = process_csv(raw_path, filename, is_input)
                # print(result)
                np.save(numpy_path + filename[:-4], result)

if __name__ == "__main__":
    convert_csvs_to_numpy()
