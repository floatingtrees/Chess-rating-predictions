# Removes unnecesary parts of the file and saves it as an array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

# Change the PGN ending to CSV so Pandas can read it
filename = "lichess_db_standard_rated_2017-05.csv"
x = pd.read_csv(filename, sep='delimiter', header=None, engine = 'python')
arr = np.array(x)
print(arr.shape)
x = 1
arr2 = np.array(arr[:arr.shape[0]//3, :])
print(arr2.shape)
p = 0
for i in range(arr.shape[0]):
	x = str(np.squeeze(arr[i]))[:7]
	if x =="[WhiteE":
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	elif x == "[BlackE":
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	elif x == '[WhiteR':
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	elif x == '[TimeCo':
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	elif x[:2] == '1.':
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	if i % 100000 == 0:
		print(i)
arr2 = arr2[:p, :]
np.save('array.npy', arr2)
