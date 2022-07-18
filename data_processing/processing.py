import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
#Turns the file into a numpy array and gets rid of unneeded parts
x = pd.read_csv("lichess_db_standard_rated_2016-08.csv", sep='delimiter', header=None, engine = 'python') # Replace with the necessary filename
arr = np.array(x)
np.save("array.npy", arr)
print(arr.shape)
x = 1
arr2 = np.array(arr[:arr.shape[0]//3, :])
print(arr2.shape)
p = 0
for i in range(arr.shape[0]):
	x = str(np.squeeze(arr[i]))[:7]
	if x =="[WhiteE":
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	elif x == "[BlackE":
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	elif x == '[WhiteR':
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	elif x == '[TimeCo':
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	elif x[:2] == '1.':
		arr2[p] = str(np.squeeze(arr[i]))
		p += 1
	if i % 100000 == 0:
		print(i)
arr = arr2[:p, :]
arr2 = 0
# Splits the array into 5 other arrays and removes incomplete data
moves = np.empty((arr.shape[0]//4), dtype=object)
diff = np.empty((arr.shape[0]//4), dtype=object)
time = np.empty((arr.shape[0]//4), dtype=object)
white = np.empty((arr.shape[0]//4), dtype=object)
black = np.empty((arr.shape[0]//4), dtype=object)
p = 0
forward = np.zeros((5,))
cache = np.empty((5,), dtype=object)
for i in range(arr.shape[0]):
	x = str(np.squeeze(arr[i]))[:7]
	if x =="[WhiteE":
		cache[0] = str(np.squeeze(arr[i]))
		forward[0] = i	
	elif x == "[BlackE":
		cache[1] = str(np.squeeze(arr[i]))
		forward[1] = i
	elif x == '[TimeCo':
		cache[3] = str(np.squeeze(arr[i]))
		forward[3] = i	
	elif x == '[WhiteR':
		cache[2] = str(np.squeeze(arr[i]))
		forward[2] = i
	elif x[:2] == '1.':
		cache[4] = str(np.squeeze(arr[i]))
		forward[4] = i
	if forward[0] + 1 == forward[1] and forward[1]+1== forward[2] and forward[2] + 1 == forward[3] and forward[3] + 1 == forward[4]:
		white[p] = str(cache[0])
		black[p] = str(cache[1])
		time[p] = str(cache[3])
		diff[p] = str(cache[2])
		moves[p] = str(cache[4])
		p += 1
	if i % 10000 == 0:
		print(i)
arr = 0
moves = moves[:p]
diff = diff[:p]
time = time[:p]
white = white[:p]
black = black[:p]
ratings = np.squeeze(np.stack((white, black), axis = 0))
# Removes turn numbers
p = 0
for game in moves:
	turns = 1
	for i in game:
		if i =='.':
			turns += 1
	for i in reversed(range(1, turns)):
		game = game.replace(f"{i}.", "")
	moves[p] = game
	if p % 10000 == 0:
		print(p)
	p += 1
# P4a
diff2 = np.zeros(diff.shape)
for i in range(len(diff)):
	for j in range(1, 4):
		x = diff[i][19:19+j]
		try:
			y = int(x)
		except ValueError:
			diff2[i] = y 
			break
diff = np.array(diff2)
# Removes games with rating changes greater than one standard deviation from the mean
p = 0
n = 0
moves2 = np.empty(moves.shape, dtype = object)
diff2 = np.zeros(diff.shape)
ratings = np.transpose(ratings)
ratings2 = np.empty(ratings.shape, dtype = object)
time2 = np.empty(time.shape, dtype = object)
try:
	for change in diff:
		if change < 19:
			moves2[p] = moves[n]
			ratings2[p, :] = ratings[n, :]
			time2[p] = time[n]
			diff2[p] = diff[n]
			p += 1
		n += 1
		if n % 10000 == 0:
			print(n)
except IndexError:
	pass
moves = moves2[:p]
ratings = ratings2[:p, :]
time = time2[:p]
diff = diff2[:p]
# Removes evaluation numbers from the games
p = 0
n = 0
try:
	for game in moves:
		game2 = game.replace("eval", "")
		if game == game2:
			moves2[p] = game2
			ratings2[p, :] = ratings[n, :]
			time2[p] = time[n]
			p += 1
		n += 1
		if n % 10000 == 0:
			print(n, p)
except IndexError:
	pass
moves = moves2[:p]
ratings = ratings2[:p, :]
time = time2[:p]
# Removes time notation from the games
p = 0
for game in moves:
	start = []
	stop = []
	for i in range(len(game)):
		if game[i] =='{':
			start.append(i)
		if game[i] =='}':
			stop.append(i)
	for i in reversed(range(len(start))):
		game = game[:start[i]] + game[stop[i]:]
	game = game.replace("}", "")
	moves[p] = game
	p += 1
	if p % 10000 == 0:
		print(p)
# Splits the data into train, validation, and test
inputs = np.stack((time, moves), axis = 1)
print(inputs.shape)
X_train, X_test, Y_train, Y_test = train_test_split(inputs, ratings, test_size = 0.1, random_state = 3)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 3)
np.save("X_train", X_train)
np.save("Y_train", Y_train)
np.save("X_val", X_val)
np.save("Y_val", Y_val)
np.save("X_test", X_test)
np.save("Y_test", Y_test)
# P6
sections = ["train", 'val', 'test']
for ttv in sections:
	x = np.load(f'Y_{ttv}.npy', allow_pickle = True)
	white = x[:, 0]
	black = x[:, 1]
	print(white)
	y = np.zeros(x.shape)
	p = 0 
	for elo in white:
		cache = []
		for i in range(len(elo)):
			try:
				x = int(elo[i])
			except ValueError:
				pass
			else:
				try:
					for j in range(5):
						int(elo[i + j])
						cache.append(elo[i+j])
				except ValueError:
					break
		connected =  int("".join(cache))
		y[p, 0] = connected
		p += 1
	p = 0
	for elo in black:
		cache = []
		for i in range(len(elo)):
			try:
				x = int(elo[i])
			except ValueError:
				pass
			else:
				try:
					for j in range(5):
						int(elo[i + j])
						cache.append(elo[i+j])
				except ValueError:
					break
		connected =  int("".join(cache))
		y[p, 1] = connected
		p += 1
	np.save(f'Y_{ttv}2.npy', y)
# Removes games that have a rating difference (between players) greater than 50
for ttv in sections:
	x = np.load(f'X_{ttv}.npy', allow_pickle = True)
	x1 = np.array(x)
	y = np.load(f'Y_{ttv}2.npy', allow_pickle = True)
	y1 = np.array(y)
	p = 0
	for i in range(len(y)):
		if abs(y[i, 0] - y[i, 1]) < 50:
			x1[p, :] = x[i, :]
			y1[p, :] = y[i, :]
			p += 1
		if i % 10000 == 0:
			print(i)
	np.save(f'Y_{ttv}3.npy', y1)
	np.save(f'X_{ttv}3.npy', x1)
# Distributes data uniformly over 1000 rating points
for ttv in sections:
	x = np.load(f'X_{ttv}3.npy', allow_pickle = True)
	x1 = np.empty(x.shape, dtype = 'object')
	y = np.load(f'Y_{ttv}3.npy', allow_pickle = True)
	y1 = np.zeros(y.shape)
	p = 0
	for i in range(len(y)):
		add = True
		if p >= 1000:
			for j in range(1000):
				if y1[p - j, 0] == y[i, 0]:
					add = False
		if add is True:
			x1[p, :] = x[i, :]
			y1[p, :] = y[i, :]
			p += 1
		if i % 10000 == 0:
			print(i, p)
	x1 = x1[:p, :]
	y1 = y1[:p, :]
	np.save(f'Y_{ttv}4.npy', y1)
	np.save(f'X_{ttv}4.npy', x1)

