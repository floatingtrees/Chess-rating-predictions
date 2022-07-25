import numpy as np
import matplotlib.pyplot as plt


adapted = np.load('X_train.npy', allow_pickle = True)
adapted = adapted[:, 1]
startw = {}
startb = {}
sw = 0
sb = 0
for game in adapted:
	game = game.replace("*", "")
	moves = game.split()
	try:
		x = startw[moves[0]]
	except:
		startw[moves[0]] = sw
		sw += 1
	try:
		x = startb[moves[1]]
	except:
		startb[moves[1]] = sb
		sb += 1
print(len(startw), startw)
print(len(startb), startb)



sections = ["test", "val", 'train']
for ttv in sections:
	print(ttv)
	X_train = np.load(f'X_{ttv}.npy', allow_pickle = True)
	#Y_train = np.load(f'Y_{ttv}.npy', allow_pickle = True)
	out = np.zeros((X_train.shape[0], 10 * 10 + 1))
	X_train = X_train[:, 1]
	p = 0
	for game in X_train:
		moves = game.split()
		w = startw[moves[0]]
		b = startb[moves[1]]
		if w < 10 and b < 10:
			out[p, w * 10 + b] = 1
		else:
			out[p, 100] = 1
		p += 1
	print(np.mean(out))
	print(np.sum(out)/out.shape[0])
	np.save(f"One_hot_{ttv}.npy", out)
	print(np.min(out), np.max(out))












