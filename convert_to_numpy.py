import os
import numpy as np

for directory in os.listdir("./raw_data"):
	if directory.startswith("."): continue
	raw_path = "./raw_data/" + directory + "/"
	numpy_path = "./numpy_data/" + directory + "/"
	for filename in os.listdir(raw_path):
		result = np.genfromtxt(raw_path + filename, delimiter=",")
		np.save(numpy_path + filename[:-4], result)

