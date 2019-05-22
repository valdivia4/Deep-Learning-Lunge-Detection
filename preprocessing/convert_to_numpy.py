import os
import scipy.io as sio
import numpy as np
import pandas as pd

def normalize(features):
    ave=np.nanmean(features, axis=0)
    #print(ave)
    std=np.nanstd(features, axis=0)
    #print(std)
    return (features-ave)/std

def clean_data(features):
	features=normalize(features)
	# print()
	df=pd.DataFrame(features)
	df = df.fillna(axis=0, method='ffill').fillna(axis=0, method='bfill')
	assert not (np.any(np.isnan(df)))
	features = np.array(df)
	return features


for directory in os.listdir("./raw_data"):
	if directory.startswith("."): continue
	raw_path = "./raw_data/" + directory + "/"
	numpy_path = "./numpy_data/" + directory + "/"
	for filename in os.listdir(raw_path):
		if filename.endswith('.txt'):
			result = pd.read_csv(raw_path + filename, sep = ',', header=None)
			result = np.array(result)
			#result = np.genfromtxt(raw_path + filename, delimiter=",")
			if filename.startswith('inputs'):
				result = clean_data(result)
			#print(result)
			np.save(numpy_path + filename[:-4], result)

