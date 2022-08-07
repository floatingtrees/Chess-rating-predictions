import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
# Tried this with 5 board positions instead of 2, which failed (more overfitting). 
model = Sequential([
	Conv2D(32, (3,3), activation='relu', input_shape=(8,8,12), padding = 'same'),
	Dropout(0.3),
	MaxPooling2D(2,2),
	Conv2D(32, (3,3), activation='relu', padding = 'same'),
	Dropout(0.3),
	MaxPooling2D(2,2),
	Conv2D(32, (3,3), activation='relu', padding = 'same'),
	Dropout(0.5),
	Flatten(),
	Dense(100, activation='relu')
])

input1 = Input((8,8,12))
input2 = Input((8,8,12))
#input3 = Input((8,8,12))
#input4 = Input((8,8,12))
#input5 = Input((8,8,12))


encoded1 = model(input1)
encoded2 = model(input2)
#encoded3 = model(input3)
#encoded4 = model(input4)
#encoded5 = model(input5)

L1_layer = Lambda(lambda tensor:(K.abs(tensor[0] - tensor[1])))
L1_distance1 = L1_layer([encoded1, encoded2])
#L1_distance2 = L1_layer([encoded2, encoded3])
#L1_distance3 = L1_layer([encoded3, encoded4])
#L1_distance4 = L1_layer([encoded4, encoded5])
#conc = layers.Concatenate(axis = -1)([L1_distance1, L1_distance2, L1_distance3, L1_distance4])
x = Dropout(0.5)(L1_distance1)
x = Dense(64, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation = 'relu')(x)
x = Dropout(0.3)(x)
prediction = Dense(1)(x)
siamese_net = Model(inputs=[input1,input2],outputs=prediction)
siamese_net.compile(loss="mean_squared_error", optimizer="Adam",metrics=["mean_absolute_error"], steps_per_execution = 1)
print(model.summary())
print(siamese_net.summary())

X_val_cat1 = np.load('X_val_cat1a.npy', allow_pickle = True)
X_val_cat2 = np.load('X_val_cat1b.npy', allow_pickle = True)
#X_val_cat3 = np.load('X_val_cat1c.npy', allow_pickle = True)
#X_val_cat4 = np.load('X_val_cat1d.npy', allow_pickle = True)
#X_val_cat5 = np.load('X_val_cat1e.npy', allow_pickle = True)
Y_val_cat = np.load('Y_val_cat1.npy', allow_pickle = True)
print(Y_val_cat.shape)
count = 0
#[X_val_cat1, X_val_cat2, X_val_cat3, X_val_cat4, X_val_cat5], Y_val_cat)
while True:
	cont = True
	count += 1
	for i in range(1, 9):
		try:
			callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = 3), 
				 tf.keras.callbacks.ModelCheckpoint(f'kitten{i}_round{count}', save_best_only = True)]
			X_train_cat1 = np.load(f'X_train_cat{i}a.npy', allow_pickle = True)
			X_train_cat2 = np.load(f'X_train_cat{i}b.npy', allow_pickle = True)
			#X_train_cat3 = np.load(f'X_train_cat{i}c.npy', allow_pickle = True)
			#X_train_cat4 = np.load(f'X_train_cat{i}d.npy', allow_pickle = True)
			#X_train_cat5 = np.load(f'X_train_cat{i}e.npy', allow_pickle = True)
			Y_train_cat = np.load(f'Y_train_cat{i}.npy', allow_pickle = True)
			siamese_net.fit([X_train_cat1, X_train_cat2], Y_train_cat, batch_size = 8,
				epochs = 100, validation_split = 0.2, 
				callbacks = callbacks, steps_per_epoch = 100000, validation_steps = 20000, use_multiprocessing = True)
		except KeyboardInterrupt:
	  		go_forward = input("\nKeep going? ")
	  		if go_forward == "N":
	  			cont = False
	  			break
	if cont is False:
	  	break
