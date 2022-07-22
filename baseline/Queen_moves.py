import numpy as np
import matplotlib.pyplot as plt


sections = ["test", "val", 'train']
for ttv in sections:
	print(ttv)
	X_train = np.load(f'X_{ttv}.npy', allow_pickle = True)
	#Y_train = np.load(f'Y_{ttv}.npy', allow_pickle = True)
	out = np.zeros((X_train.shape[0], 2))
	X_train = X_train[:, 1]
	p = 0
	for game in X_train:
		moves = game.split()
		queen_moves = 0
		for i in moves:
			if i[0] == "Q":
				queen_moves += 1
		out[p, 0] = queen_moves
		out[p, 1] = len(moves)
		p += 1
	np.save(f"Queen_moves_{ttv}", out)
