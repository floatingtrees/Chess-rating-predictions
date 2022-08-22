import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

# Create a folder call people_groups
x = pd.read_csv("testing.csv", sep='delimiter', header=None, engine = 'python', on_bad_lines='warn')
print("loaded")
arr = np.array(x)
np.save("array.npy", x)
x = 0 
#arr = np.array(arr[:arr.shape[0]//100, :])
p = 0
names_pre = {}
for i in range(arr.shape[0]):
	text = str(np.squeeze(arr[i]))
	if text[:8] =='[White "':
		try:
			count = names_pre[text]
		except KeyError:
			names_pre[text] = 1 
		else:
			names_pre[text] = count + 1
print('keys_created')
games = {}
high = 0
names = {}
for key in names_pre:
	x = names_pre[key]
	if x > high:
		high = x
	if x >= 30: # choose how many games you want
		names[key] = x
print("low_numbers_deleted", high)

count = 0
for key in names:
	count += 1
	place = 0
	for i in range(arr.shape[0]):
		text = str(np.squeeze(arr[i]))
		if text == key:
			if place == 0:
				placeholder_array = np.empty((high * 16, 1), dtype = object)
			try:
				placeholder_array[place * 16:(place + 1) * 16] = arr[i: i + 16]
			except ValueError:
				continue
			place += 1
	placeholder_array = placeholder_array[:place * 16]
	np.save(f"people_groups/multiperson{count}.npy", placeholder_array)
	print(placeholder_array.shape)
