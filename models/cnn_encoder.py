import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, Reshape, Flatten)

tfd = tfp.distributions

class CNNEncoder(tf.keras.Model):

	def __init__(self, args):
		super(CNNEncoder).__init__()
		self.encoder_size = args.encoder_size
		self._build_model()

	def _build_model(self):
		if self.model_size=='small':
            params=[16, 3, 32, 3, 256]
        if self.model_size=='large':
            params=[32, 5, 64, 5, 2048]

        self.reshape = Reshape((28,28,1), input_shape=(784,))
        self.conv1 = Conv2D(filters=params[0], kernel_size=params[1],
            padding='same', activation='relu')
        self.pool1 = MaxPooling2D(2)
        self.conv2 = Conv2D(filters=params[2], kernel_size=params[3],
            padding='same', activation='relu')
        self.pool2 = MaxPooling2D(2)
        self.flatten = Flatten()
        self.layer1 = Dense(units=params[4], activation='relu')
        self.out = Dense(62, activation='softmax')
		pass

	def call(self, x, y):
		pass

