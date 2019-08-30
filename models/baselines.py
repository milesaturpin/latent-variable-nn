import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, Reshape, Flatten)
#from tensorflow_probability.layers import DenseVariational

from models.base_model import BaseModel
from models.model_utils import (
    init_mean_field_vectors, build_normal_variational_posterior,
    latent_normal_matrix, latent_matrix_variational_posterior, softplus_inverse)

from tensorflow.keras.utils import to_categorical

from models.multilevel_layers import MultilevelDense, FactoredMultilevelDense

tfd = tfp.distributions
tfpl = tfp.layers

class MLP(BaseModel):

    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)

    def _build_model(self):
        self.dense1 = Dense(units=512, activation='relu')
        self.dense2 = Dense(units=256, activation='relu')
        self.out = Dense(62, activation='softmax')

    def call(self, x, gid):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class OneHotMLP(BaseModel):

    def __init__(self, *args, **kwargs):
        super(OneHotMLP, self).__init__(*args, **kwargs)

    def _build_model(self):
        self.dense1 = Dense(units=512, activation='relu')
        self.dense2 = Dense(units=256, activation='relu')
        self.out = Dense(62, activation='softmax')

    def call(self, x, gid):
        inputs = np.concatenate([x, to_categorical(gid, num_classes=self.num_groups[0])], axis=1)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x


class MultilevelMLP(BaseModel):

    def __init__(self, *args, **kwargs):
        super(MultilevelMLP, self).__init__(*args, **kwargs)

    def _build_model(self):
        self.dense1 = Dense(units=512, activation='relu')
        self.dense2 = Dense(units=256, activation='relu')
        self.ml_dense = MultilevelDense(62, num_groups=self.num_groups[0], activation='softmax')

    def call(self, x, gid):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.ml_dense(x, gid)
        return x


class FactoredMultilevelMLP(BaseModel):

    def __init__(self, *args, **kwargs):
        super(FactoredMultilevelMLP, self).__init__(*args, **kwargs)

    def _build_model(self):
        self.dense1 = Dense(units=512, activation='relu')
        self.dense2 = Dense(units=256, activation='relu')
        self.ml_dense = FactoredMultilevelDense(
            units=62,
            num_groups=self.num_groups[0],
            multilevel_weights=True,
            multilevel_bias=True,
            weights_latent_dim=self.z_dim[0],
            bias_latent_dim=self.z_dim[1],
            activation='softmax')

    def call(self, x, gid):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.ml_dense(x, gid)
        return x



