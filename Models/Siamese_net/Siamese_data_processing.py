import numpy as np
from tensorflow import keras
import tensorflow as tf 
from keras import layers
from tensorflow.keras import backend as K

def cat_split(X_train1, Y_train1):
  count = 0
  X_train_cat1 = np.zeros((1250000, 8, 8, 12))
  X_train_cat2 = np.zeros((1250000, 8, 8, 12))
  Y_train_cat = np.zeros((1250000))
  for i in range(X_train1.shape[0]):
    if count >= 1249850:
        break
    for j in range(20, 149, 2):
      pre1 = X_train1[i, :, :, :, j]
      pre2 = X_train1[i, :, :, :, j+1]
      if np.sum(pre2) == 0:
        break
      else:
        X_train_cat1[count, :, :, :] = np.reshape(pre1, (1, 8, 8, 12))
        X_train_cat2[count, :, :, :] = np.reshape(pre2, (1, 8, 8, 12))
        Y_train_cat[count] = Y_train1[i]
        count += 1
  X_train_cat1 = X_train_cat1[:count, :, :, :]
  X_train_cat2 = X_train_cat2[:count, :, :, :]
  Y_train_cat = Y_train_cat[:count]
  return X_train_cat1, X_train_cat2, Y_train_cat

split = 'train'
for i in range(1, 11):
    X_train = np.load(f'X_{split}_board{i}a.npy', allow_pickle = True)
    Y_train = np.load(f'Y_{split}_board{i}a.npy', allow_pickle = True)
    X1, X2, Y = cat_split(X_train, Y_train)
    np.save(f"X_{split}_cat{i}a.npy", X1)
    np.save(f"X_{split}_cat{i}b.npy", X2)
    np.save(f"Y_{split}_cat{i}.npy", Y)
#X_test = np.load('X_test_board1a.npy', allow_pickle = True)
#Y_test = np.load('Y_test_board1a.npy', allow_pickle = True)
