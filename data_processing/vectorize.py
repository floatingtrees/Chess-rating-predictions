import numpy as np
from tensorflow import keras
import tensorflow as tf 
from keras import layers

X_train1 = np.load('X_train_proper.npy', allow_pickle = True)
Y_train1 = np.load('Y_train_proper.npy', allow_pickle = True)
print(Y_train1.shape)
X_val1 = np.load('X_val_proper.npy', allow_pickle = True)
Y_val1 = np.load('Y_val_proper.npy', allow_pickle = True)
X_test1 = np.load('X_test_proper.npy', allow_pickle = True)
Y_test1 = np.load('Y_test_proper.npy', allow_pickle = True).astype(np.float32)
sequence_length = 300
print(Y_train1)
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
	max_tokens= 20000,
	standardize=None,
	split="whitespace",
	ngrams=None,
	output_mode="int",
	output_sequence_length=sequence_length,
	pad_to_max_tokens=True,
	vocabulary=None
)
vectorize_layer.adapt(X_train1)
print("A")
X_train  = vectorize_layer(X_train1)
print("A")
X_val = vectorize_layer(X_val1)
X_test = vectorize_layer(X_test1)

np.save("X_train_vect1.npy", X_train.numpy())
np.save("X_val_vect1.npy", X_val.numpy())
np.save("X_test_vect1.npy", X_test.numpy())










