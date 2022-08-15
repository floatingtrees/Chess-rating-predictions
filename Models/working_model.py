import numpy as np
from tensorflow import keras
import tensorflow as tf 
from keras import layers

sequence_length = 300
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
embed_dims = 32
dropout_rate = 0.2
inputs1 = keras.Input(shape = (sequence_length,)) #baseline: 178.7 
embed = PositionalEmbedding(sequence_length, 20001, embed_dims)(inputs1)
x = layers.Flatten()(embed)
x = layers.Dropout(dropout_rate)(x) #
x = layers.Dense(embed_dims, activation = 'relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(dropout_rate)(x) #
x = layers.Dense(embed_dims, activation = 'relu')(x)
x = layers.Dropout(dropout_rate)(x) #
output = layers.Dense(1)(x)
model = keras.Model(inputs1, output)
model.compile(keras.optimizers.Adam(0.01), 'MSE', ["mean_absolute_error"], steps_per_execution = 1)
print(model.summary())

def convert(x):
	return tf.cast(x, dtype = tf.float32)

X_train = np.load('X_train_vect.npy', allow_pickle = True)
Y_train = np.load('Y_train_proper.npy', allow_pickle = True)

X_val = np.load('X_val_vect.npy', allow_pickle = True)
Y_val = np.load('Y_val_proper.npy', allow_pickle = True)

X_test = np.load('X_test_vect.npy', allow_pickle = True)
Y_test = np.load('Y_test_proper.npy', allow_pickle = True)
Y_train, Y_val, Y_test = np.mean(Y_train, axis = -1), np.mean(Y_val, axis = -1), np.mean(Y_test, axis = -1)


callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)]
# 415,561 steps ~6 hours
print(X_train.shape)

num_examples = 790000
try:
	model.fit(x = X_train[:48000, :], y = Y_train[:48000], epochs = 3, 
			validation_data = (X_val, Y_val), callbacks = callbacks, 
			batch_size = 16, steps_per_epoch = 1000, validation_steps = None, validation_freq = 3)
	model.fit(x = X_train[48000:num_examples, :], y = Y_train[48000:num_examples], epochs = 8, 
		validation_data = (X_val, Y_val), callbacks = callbacks, 
		batch_size = 256, steps_per_epoch = None, validation_steps = None, validation_freq = 1)
except KeyboardInterrupt:
	pass
finally:
	model.save("smol_model_half", save_format = 'h5')
	print("\n")
	print("PRED:")
	print(model.predict(X_test[:20, :]))
	print(Y_test[:20])
	print(model.evaluate(X_test, Y_test))














