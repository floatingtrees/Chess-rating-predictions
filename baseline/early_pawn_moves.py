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
		pawn_movesw = 0
		pawn_movesb = 0
		even = 0
		for i in moves:
			even += 1
			if i[0] != "Q" and i[0] != "R" and i[0] != "N" and i[0] != "B" and i[0] != "K":
				if even % 2 == 1:
					pawn_movesw += 1
				else:
					pawn_movesb += 1
			if even >= 30:
				break
		out[p, 0] = pawn_movesw
		out[p, 1] = pawn_movesb
		p += 1
	print(np.max(out))
	np.save(f"Pawn_moves_{ttv}", out)
