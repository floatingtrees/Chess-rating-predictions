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
		game = game.replace("*", "")
		moves = game.split()
		even = 0
		wt = 0
		bt = 0
		for i in moves:
			even += 1
			try:
				if i[1] == "x":
					if even % 2 == 1:
						wt += 1
					else:
						bt += 1
			except IndexError:
				print(i, '\n')
				print(moves)
		out[p, 0] = wt
		out[p, 1] = bt
		p += 1
	np.save(f"Captures_{ttv}.npy", out)
	print(np.min(out), np.max(out))
