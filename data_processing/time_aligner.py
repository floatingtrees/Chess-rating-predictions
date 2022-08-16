import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

# Makes sure that all games used are 10 minutes long. 
sections = ["val", 'train', 'test']
for ttv in sections:
	x = np.load(f'X_{ttv}.npy', allow_pickle = True)
	x1 = np.empty((x.shape[0]), dtype = 'object')
	y = np.load(f'Y_{ttv}.npy', allow_pickle = True)
	y1 = np.zeros(y.shape)
	p = 0
	for i in range(len(y)):
		proceed = False
		if x[i, 0] == '[TimeControl "600+0"]':
			proceed = True 
		if proceed is True:
			x1[p] = x[i, 1]
			y1[p] = y[i, 0]
			p += 1 
	x1 = x1[:p]
	print(x1)
	y1 = y1[:p]
	np.save(f'Y_{ttv}_proper.npy', y1)
	np.save(f'X_{ttv}_proper.npy', x1)
