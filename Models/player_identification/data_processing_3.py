import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import sys

# this is the last one
np.set_printoptions(threshold = 100000)
rng = np.random.default_rng(seed = 0)
X_train_dict = {}
Y_train_dict = {}
X_val_dict = {}
Y_val_dict = {}
for i in range(1,197):
	X = np.load(f'people_processed_2016-3/X_multiperson_vect{i}.npy', allow_pickle = True)
	Y = np.load(f"people_processed_2016-3/Y_multiperson{i}.npy")
	Y = np.zeros((8, 1))
	Y.fill(i)
	X = X[:8, :]
	X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state = i, test_size = 0.25)
	X_train_dict[str(i)] = X_train
	Y_train_dict[str(i)] = Y_train
	X_val_dict[str(i)] = X_val
	Y_val_dict[str(i)] = Y_val

for key in X_train_dict:
	if key == '1':
		X_train_save1 = X_train_dict[key]
		X_train_save1b = rng.permutation(X_train_save1)
		Y_train_save = Y_train_dict[key]
		X_val_save1 = X_val_dict[key]
		X_val_save1b = rng.permutation(X_val_save1)
		Y_val_save = Y_val_dict[key]
	else:
		X_train_save1 = np.concatenate((X_train_save1, X_train_dict[key]), axis = 0)
		X_train_save1b = np.concatenate((X_train_save1b, rng.permutation(X_train_dict[key])), axis = 0)
		Y_train_save = np.concatenate((Y_train_save, Y_train_dict[key]), axis = 0)
		X_val_save1 = np.concatenate((X_val_save1, X_val_dict[key]), axis = 0)
		X_val_save1b = np.concatenate((X_val_save1b, rng.permutation(X_val_dict[key])), axis = 0)
		Y_val_save = np.concatenate((Y_val_save, Y_val_dict[key]), axis = 0)

while True:
	seed = int(random.random() * 1000000)
	rng = np.random.default_rng(seed = seed)
	X_val2 = rng.permutation(X_val_save1)
	rng = np.random.default_rng(seed = seed)
	Y_val2 = rng.permutation(Y_val_save)
	if np.all(Y_val2 - Y_val_save) != 0:
		break
print('val_done')

while True:
	seed = int(random.random() * 1000000)
	rng = np.random.default_rng(seed = seed)
	X_train2 = np.random.permutation(X_train_save1)
	rng = np.random.default_rng(seed = seed)
	Y_train2 = np.random.permutation(Y_train_save)
	if np.all(Y_train2 - Y_train_save) != 0:
		break
good = True
for i in range(Y_train2.shape[0]):
	if good is False:
		print("failed")
		sys.exit()
	good = False
	label = Y_train_save[i, 0]
	game = X_train_save1[i, 0]
	for j in range(Y_val2.shape[0]):
		if Y_train2[j, 0] == label and X_train2[j, 0] == game:
			print("found")
			good = True
			break

np.save("X_train_ID1a", X_train_save1)
np.save("X_train_ID1b", X_train_save1b)
np.save('X_train_ID2', X_train2)
np.save("Y_train_ID", Y_train_save)
np.save("X_val_ID1a", X_val_save1)
np.save("X_val_ID1b", X_val_save1b)
np.save('X_val_ID2', X_val2)
np.save("Y_val_ID", Y_val_save)
