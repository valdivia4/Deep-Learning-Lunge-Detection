import os
import numpy as np

for filename in os.listdir(os.curdir):
	if filename.endswith(".npy"):
		array = np.load(filename)
		np.savetxt(filename[:-4] + ".txt", array)

