import numpy as np
import matplotlib.pyplot as plt


sections = ["test", "val", 'train']
np.set_printoptions(threshold = 1000)
for ttv in sections:
	print(ttv)
	X_train = np.load(f'X_{ttv}.npy', allow_pickle = True)
	#Y_train = np.load(f'Y_{ttv}.npy', allow_pickle = True)
	out = np.zeros((X_train.shape[0], 3))
	X_train = X_train[:, 1]
	p = 0
	for game in X_train:
		game = game.replace("*", "")
		moves = game.split()
		result = moves[-1]
		if result[0] == '1':
			if result[1] == '/':
				out[p, 0] = 1
			else:
				out[p, 1] = 1
		else:
			out[p, 2] = 1
		p += 1
	print(np.mean(out))
	print(out[:, 0])
	np.save(f"Winner_{ttv}.npy", out)
	print(np.min(out), np.max(out))
