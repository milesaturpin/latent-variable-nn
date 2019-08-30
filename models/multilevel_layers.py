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

tfd = tfp.distributions
tfpl = tfp.layers


class MultilevelDense(tf.keras.layers.Layer):

    def __init__(self, units, num_groups, multilevel_weights=False,
             multilevel_bias=True, fixed_prior=True, activation=None, **kwargs):
        super(MultilevelDense, self).__init__(**kwargs)
        self.units = units
        self.num_groups = num_groups
        self.multilevel_weights = multilevel_weights
        self.multilevel_bias = multilevel_bias
        self.fixed_prior = fixed_prior
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):

        # self.multi_w_mu = self.add_weight(
        #     shape=(self.num_groups, input_shape[-1], self.units),
        #     initializer='random_normal',
        #     trainable=True)
        # self.multi_b_mu = self.add_weight(
        #     shape=(self.num_groups, self.units),
        #     initializer='zeros',
        #     trainable=True)

        # Note the multiplication of the last 2 terms
        if self.multilevel_weights:
            weight_rvs = init_mean_field_vectors(
                shape=(self.num_groups, input_shape[-1] * self.units),
                fixed_prior=self.fixed_prior)

            if self.fixed_prior:
                self.w_mu, self.w_sigma, self.w_prior = weight_rvs
            else:
                (self.w_mu, self.w_sigma, self.w_prior,
                 self.w0_mu, self.w0_sigma, self.w0_prior) = weight_rvs
        else:
            #self.w = Dense(self.units)
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='random_normal',
                trainable=True)
            #self.w = tf.Variable(tfd.Normal(0,0.1).sample([input_shape[-1], self.units]))

        if self.multilevel_bias:
            bias_rvs = init_mean_field_vectors(
                shape=(self.num_groups, self.units),
                fixed_prior=self.fixed_prior,
                mu_initializer='zeros')

            if self.fixed_prior:
                self.b_mu, self.b_sigma, self.b_prior = bias_rvs
            else:
                (self.b_mu, self.b_sigma, self.b_prior,
                 self.b0_mu, self.b0_sigma, self.b0_prior) = bias_rvs
        else:
            self.b = self.add_weight(
                shape=(self.units),
                initializer='zeros',
                trainable=True)
            #self.b = tf.Variable(np.zeros((self.units), dtype=np.float32))

        super(MultilevelDense, self).build(input_shape)

    def construct_posterior(self, z_mu, z_sigma, z_prior, gid):
        """Build the posterior and compute the loss."""

        z_post = build_normal_variational_posterior(
            z_mu, z_sigma, gid)

        kl_loss = tfd.kl_divergence(z_post, z_prior)
        self.add_loss(tf.reduce_sum(kl_loss))
        return z_post

    def call(self, x, gid):

        if self.multilevel_weights:
            w_post = self.construct_posterior(
                self.w_mu, self.w_sigma, self.w_prior, gid)
            w = Reshape((-1, self.units))(w_post.sample())
        else:
            w = self.w

        if self.multilevel_bias:
            b_post = self.construct_posterior(
                self.b_mu, self.b_sigma, self.b_prior, gid)
            b = b_post.sample()
        else:
            b = self.b

        x = tf.expand_dims(x, axis=-1)
        x = tf.matmul(w, x, transpose_a=True)
        x = tf.squeeze(x)
        out = self.activation(x + b)

        return out

    def get_config(self):
        config = super(MultilevelDense, self).get_config()
        config.update({'units': self.units, 'num_groups': self.num_groups,
            'multilevel_weights' : False, 'multilevel_bias' : True,
            'fixed_prior' : True, 'activation' : None})
        return config



class FactoredMultilevelDense(tf.keras.layers.Layer):

    def __init__(self, units, num_groups, multilevel_weights=True, multilevel_bias=True,
                 weights_latent_dim=None, bias_latent_dim=None, fixed_prior=True, activation=None, **kwargs):
        super(FactoredMultilevelDense, self).__init__(**kwargs)
        self.units = units
        self.num_groups = num_groups
        self.multilevel_weights = multilevel_weights
        self.multilevel_bias = multilevel_bias
        self.weights_latent_dim = weights_latent_dim
        self.bias_latent_dim = bias_latent_dim
        self.fixed_prior = fixed_prior
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):

        if self.multilevel_weights:

            # Use latent dimension if provided, or else model full parameters
            if self.weights_latent_dim is not None:
                rv_dim = self.weights_latent_dim
                self.weight_factors = Dense(input_shape[-1] * self.units, use_bias=False)
            else:
                rv_dim = input_shape[-1] * self.units

            weight_rvs = init_mean_field_vectors(
                shape=(self.num_groups, rv_dim),
                fixed_prior=self.fixed_prior)
            self.reshape_weights = Reshape((input_shape[-1], self.units))

            if self.fixed_prior:
                self.zw_mu, self.zw_sigma, self.zw_prior = weight_rvs
            else:
                (self.zw_mu, self.zw_sigma, self.zw_prior,
                 self.zw0_mu, self.zw0_sigma, self.zw0_prior) = weight_rvs
        else:
            #self.w = Dense(self.units)
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='random_normal',
                trainable=True)
            #self.w = tf.Variable(tfd.Normal(0,0.1).sample([input_shape[-1], self.units]))

        if self.multilevel_bias:

            # Use latent dimension if provided, or else model full parameters
            if self.bias_latent_dim is not None:
                rv_dim = self.bias_latent_dim
                self.bias_factors = Dense(self.units, use_bias=False)
            else:
                rv_dim = self.units

            bias_rvs = init_mean_field_vectors(
                shape=(self.num_groups, rv_dim),
                fixed_prior=self.fixed_prior,
                mu_initializer='zeros')

            if self.fixed_prior:
                self.zb_mu, self.zb_sigma, self.zb_prior = bias_rvs
            else:
                (self.zb_mu, self.zb_sigma, self.zb_prior,
                 self.zb0_mu, self.zb0_sigma, self.zb0_prior) = bias_rvs
        else:
            self.b = self.add_weight(
                shape=(self.units),
                initializer='zeros',
                trainable=True)
            #self.b = tf.Variable(np.zeros((self.units), dtype=np.float32))
        super(FactoredMultilevelDense, self).build(input_shape)


    def construct_posterior(self, z_mu, z_sigma, z_prior, gid):
        """Build the posterior and compute the loss."""

        z_post = build_normal_variational_posterior(
            z_mu, z_sigma, gid)

        kl_loss = tfd.kl_divergence(z_post, z_prior)
        self.add_loss(tf.reduce_sum(kl_loss))
        return z_post


    def call(self, x, gid):

        if self.multilevel_weights:
            # Sample from variational posterior over latent vars or weights
            zw_post = self.construct_posterior(
                self.zw_mu, self.zw_sigma, self.zw_prior, gid)
            zw_samp = zw_post.sample()

            # Reconstruct low rank approximation
            if self.weights_latent_dim is not None:
                zw_samp = self.weight_factors(zw_samp)

            # Reassemble into weight matrix format
            w = self.reshape_weights(zw_samp)
        else:
            w = self.w

        if self.multilevel_bias:
            # Sample from variational posterior over latent vars or weights
            zb_post = self.construct_posterior(
                self.zb_mu, self.zb_sigma, self.zb_prior, gid)
            b = zb_post.sample()

            # Reconstruct low rank approximation
            if self.bias_latent_dim is not None:
                b = self.bias_factors(b)
        else:
            b = self.b


        x = tf.expand_dims(x, axis=-1)
        x = tf.matmul(w, x, transpose_a=True)
        x = tf.squeeze(x)
        out = self.activation(x + b)

        return out

    def get_config(self):
        config = super(FactoredMultilevelDense, self).get_config()
        config.update({'units': self.units, 'num_groups': self.num_groups,
            'multilevel_weights' : False, 'multilevel_bias' : True,
            'fixed_prior' : True, 'activation' : None})
        return config


