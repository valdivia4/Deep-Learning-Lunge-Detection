import os

import numpy as np
import pandas as pd


def normalize(features):
    """
    Normalizes the deployment given by features (so that each
    feature has 0 mean 1 standard deviation)
    """

    ave = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0)
    return (features - ave) / std


def clean_data(features):
    """Normalizes the features and removes NaNs"""

    features = normalize(features)
    df = pd.DataFrame(features)
    df = df.fillna(axis=0, method='ffill').fillna(axis=0, method='bfill')
    assert not (np.any(np.isnan(df)))
    features = np.array(df)
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
