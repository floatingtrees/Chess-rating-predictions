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
		minorw = 0
		minorb = 0
		even = 0
		for i in moves:
			even += 1
			if i[0] == "B" or i[0] == "N":
				if even % 2 == 1:
					minorw += 1
				else:
					minorb += 1
			if even >= 30:
				break
		out[p, 0] = minorw
		out[p, 1] = minorb
		p += 1
	print(np.max(out), np.mean(out), np.std(out))
	np.save(f"Minors_{ttv}.npy", out)
