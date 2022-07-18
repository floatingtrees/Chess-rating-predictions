import numpy as np
from tensorflow import keras
import tensorflow as tf 
from keras import layers

def convert(x):
	return tf.cast(x, dtype = tf.float32)

X_train1 = np.load('X_train.npy', allow_pickle = True)
Y_train1 = np.load('Y_train.npy', allow_pickle = True)
X_val1 = np.load('X_val.npy', allow_pickle = True)
Y_val1 = np.load('Y_val.npy', allow_pickle = True)
X_test1 = np.load('X_test.npy', allow_pickle = True)
Y_test1 = np.load('Y_test.npy', allow_pickle = True)
X_train, X_val, X_test = X_train1[:, 1], X_val1[:10000, 1], X_test1[:2000, 1] # gets the moves only
T_train, T_val, T_test = X_train1[:, 0], X_val1[:10000, 0], X_test1[:2000, 0] # time control
Y_train, Y_val, Y_test = convert(Y_train1[:, 0]), convert(Y_val1[:10000, 0]), convert(Y_test1[:2000, 0])
sequence_length = 200

vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
	max_tokens=2500,
	standardize=None,
	split="whitespace",
	ngrams=None,
	output_mode="int",
	output_sequence_length=sequence_length,
	pad_to_max_tokens=True,
	vocabulary=None
)
time_vect = tf.keras.layers.StringLookup(
    max_tokens=None,
    num_oov_indices=1,
    mask_token=None,
    oov_token="[UNK]",
    vocabulary=None,
    idf_weights=None,
    encoding=None,
    invert=False,
    output_mode="int",
    sparse=False,
    pad_to_max_tokens=False,
)
time_vect.adapt(T_train[:200000])
T_train, T_val, T_test = time_vect(T_train), time_vect(T_val), time_vect(T_test)
vectorize_layer.adapt(X_train[:500000])
X_train  = vectorize_layer(X_train)
X_val = vectorize_layer(X_val)
X_test = vectorize_layer(X_test)
 
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
embed_dims = 128
embed_dims2 = 8
dropout_rate = 0.0
inputs1 = keras.Input(shape = (sequence_length,)) #baseline: 178.7 
inputs2 = keras.Input(shape = (1,))
#word_embeddings = keras.layers.Embedding(5000, embed_dims, mask_zero = True)(inputs)
embed = PositionalEmbedding(sequence_length, 2505, embed_dims)(inputs1)
head1 = keras.layers.MultiHeadAttention(3, key_dim = embed_dims, kernel_regularizer=keras.regularizers.L1L2(l1=0.000, l2=0.000))(embed, embed)
conc1 = keras.layers.Concatenate(axis = -1)([head1, embed])
norm1 = keras.layers.LayerNormalization()(conc1)
drop1 = keras.layers.Dropout(dropout_rate)(norm1) #
dense1 = keras.layers.Dense(embed_dims, activation = 'relu')(drop1)
dense1 = keras.layers.Dropout(dropout_rate )(dense1) #
dense2 = keras.layers.Dense(embed_dims)(dense1)
normi = layers.LayerNormalization()(dense2)
conc2 = keras.layers.Concatenate(axis = -1)([norm1, dense2])
norm2 = keras.layers.LayerNormalization()(conc2)
transpool = layers.MaxPooling1D(25, 25)(norm2)

conv1 = layers.Conv1D(32, 3, padding = 'causal')(norm2)
conv2 = layers.Conv1D(64, 3, padding = 'causal')(conv1)
pool1 = layers.MaxPooling1D(5, 5)(conv2)
conv3 = layers.Conv1D(128, 3, padding = 'causal')(pool1)
conv4 = layers.Conv1D(128, 3, padding = 'causal')(conv3)
pool2 = layers.MaxPooling1D(5, 5)(conv4)
conv5 = layers.Conv1D(128, 3, padding = 'causal')(pool2)
conv6 = layers.Conv1D(128, 3, padding = 'causal')(conv5)
normk = layers.BatchNormalization()(conv6)
embed2 = tf.transpose(layers.Embedding(1000, embed_dims2, mask_zero = True)(inputs2), (0, 2, 1))
normj = layers.BatchNormalization()(embed2)

x = layers.Concatenate(axis = -1)([normj, normk, transpool])
flatten = layers.Flatten()(x)
x = layers.Dense(embed_dims * 2, activation = 'relu')(flatten)
x = layers.Dense(embed_dims, activation = 'relu')(x)
x = layers.Dropout(dropout_rate)(x) #
x = layers.Dense(embed_dims, activation = 'relu')(x)
x = layers.Concatenate(axis = -1)([flatten, x])
x = layers.Dense(embed_dims, activation = 'relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(dropout_rate/2)(x) #
x = layers.Dense(embed_dims, activation = 'relu')(x)
x = layers.Dropout(dropout_rate/2)(x) #
output = layers.Dense(1)(x)
model = keras.Model([inputs1, inputs2], output)
model.compile(keras.optimizers.Adam(0.01), 'MSE', ["mean_absolute_error"], steps_per_execution = 4)
print(model.summary())

#train = tf.data.Dataset.from_tensor_slices(([X_train, T_train], Y_train)).batch(8)
#val = tf.data.Dataset.from_tensor_slices(([X_val, T_val], Y_val)).batch(8)
#test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(8)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5), 
			tf.keras.callbacks.ModelCheckpoint("basic_model3", monitor = 'val_loss', save_best_only = True)]
      
model.fit(x = [X_train, T_train], y = Y_train, epochs = 500, 
		validation_data = ([X_val, T_val], Y_val), callbacks = callbacks, 
		batch_size = 8, steps_per_epoch = None, validation_steps = None)













