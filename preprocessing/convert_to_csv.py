import os
import numpy as np

for filename in os.listdir("./temp"):
	if filename.endswith(".npy"):
		array = np.load("./temp/" + filename)
		np.savetxt(filename[:-4].lower() + ".txt", array, delimiter=',')
  
