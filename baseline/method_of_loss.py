import numpy as np
import matplotlib.pyplot as plt
import sys


sections = ["test", "val", 'train']
for ttv in sections:
	print(ttv)
	X_train = np.load(f'X_{ttv}.npy', allow_pickle = True)
	#Y_train = np.load(f'Y_{ttv}.npy', allow_pickle = True)
	out = np.zeros((X_train.shape[0], 2))
	X_train = X_train[:, 1]
	p = 0
	for game in X_train:
		game = game.replace("*", "")
		moves = game.split()
		result = moves[-1]
		last = moves[-2]
		if last[-1] == '#':
			out[p, 0] = 0
		else:
			out[p, 0] = 1
		if result == '1-0' or result == '0-1':
			out[p, 1] = 0
		else:
			out[p, 1] = 1
		p += 1
	print(np.max(out), np.mean(out), np.std(out))
	np.save(f"Loss_{ttv}.npy", out)
