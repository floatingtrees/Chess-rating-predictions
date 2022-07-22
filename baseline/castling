import numpy as np
import matplotlib.pyplot as plt


sections = ["test", "val", 'train']
for ttv in sections:
	print(ttv)
	X_train = np.load(f'X_{ttv}.npy', allow_pickle = True)
	#Y_train = np.load(f'Y_{ttv}.npy', allow_pickle = True)
	out = np.zeros((X_train.shape[0], 2))
	out.fill(300)
	X_train = X_train[:, 1]
	p = 0
	for game in X_train:
		moves = game.split()
		even = 0
		for i in moves:
			even += 1
			if i[0] == "O":
				if even % 2 == 1:
					out[p, 0] = even
				else:
					out[p, 1] = even
		p += 1
	np.save(f"Castle_{ttv}.npy", out)
