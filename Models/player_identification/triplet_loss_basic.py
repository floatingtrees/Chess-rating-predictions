import numpy as np
from tensorflow import keras
import tensorflow as tf 
from keras import layers
import tensorflow_addons as tfa
# basic triplet loss model
np.set_printoptions(threshold = np.inf)

class PositionalEmbedding(layers.Layer):
	def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
		super().__init__(**kwargs)
		self.token_embeddings = layers.Embedding(
			input_dim=input_dim, output_dim=output_dim)
		self.position_embeddings = layers.Embedding(
			input_dim=sequence_length, output_dim=output_dim)
		self.sequence_length = sequence_length
		self.input_dim = input_dim
		self.output_dim = output_dim
  
	def call(self, inputs):
		length = tf.shape(inputs)[-1]
		positions = tf.range(start=0, limit=length, delta=1)
		embedded_tokens = self.token_embeddings(inputs)
		embedded_positions = self.position_embeddings(positions)
		return embedded_tokens + embedded_positions
 
	def compute_mask(self, inputs, mask=None):
		return tf.math.not_equal(inputs, 0)  
 
	def get_config(self):
		config = super().get_config()
		config.update({
			"output_dim": self.output_dim,
			"sequence_length": self.sequence_length,
			"input_dim": self.input_dim,
		})
		return config



dropout_rate = 0.05
base_input = keras.Input((300, ))
x = PositionalEmbedding(300, 20005, 128)(base_input)
x = layers.Flatten()(x)
x = layers.Dropout(dropout_rate)(x) #
x = layers.Dense(512, activation = 'relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(dropout_rate)(x) #
x = layers.Dense(256, activation = 'relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(dropout_rate)(x) #
encoding = layers.Dense(128)(x)
base_model = keras.Model(base_input, encoding)

anchor = keras.Input((300, ))
positive = keras.Input((300, ))
negative = keras.Input((300, ))

ae = base_model(anchor)
pe = base_model(positive)
ne = base_model(negative)

#L2_positive = L2_layer([ae, pe])
#L2_negative = L2_layer([ae, ne])


triplet = keras.Model([anchor, positive, negative], [ae, pe, ne])
triplet.compile('adam', tfa.losses.TripletSemiHardLoss())


Xv_train1a = np.load("X_train_ID1a.npy", allow_pickle = True)
Xv_train1b = np.load("X_train_ID1b.npy", allow_pickle = True)
Xv_train2 = np.load("X_train_ID2.npy", allow_pickle = True)
X_train1a = Xv_train1a[:1000, :]
X_train1b = Xv_train1b[:1000, :]
X_train2 = Xv_train2[:1000, :]
X_val1a = Xv_train1a[1000:, :]
X_val1b = Xv_train1b[1000:, :]
X_val2 = Xv_train2[1000:, :]



filler = np.zeros(X_train1a.shape[0])
filler_val = np.zeros(X_val1a.shape[0])
triplet.fit([X_train1a, X_train1b, X_train2], [filler, filler, filler], epochs = 10,
	validation_data = ([X_val1a, X_val1b, X_val2], [filler_val, filler_val, filler_val]))
ae, pe, be = triplet.predict([X_val1a, X_val1b, X_val2])
print(np.mean(np.square(ae - pe)), np.mean(np.square(ae - be)))
triplet.save("triplet2", save_format = 'h5')
