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

class MyDense(tf.keras.layers.Layer):

    def __init__(self, units, activation=None, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        x_input_shape, gid_input_shape = input_shape

        self.w = self.add_weight(
            shape=(x_input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True)
        #self.w = tf.Variable(tfd.Normal(0,0.1).sample([x_input_shape[-1], self.units]))


        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True)
        #self.b = tf.Variable(np.zeros((self.units), dtype=np.float32))

        super(MyDense, self).build(input_shape)

    def call(self, inputs):
        x, gid = inputs

        batch_size, num_features = x.shape
        assert len(x.shape) >= 2, "Data is incorrect shape!"
        assert len(gid.shape) == 1, "gid should be flat vector!"

        w = self.w
        b = self.b

        # B: batch size, p: num_features, u: num_units
        einsum_matrix_mult = 'pu,Bp->Bu'.format()
        x = tf.einsum(einsum_matrix_mult, w, x)

        target_shape = (batch_size, self.units)
        msg = "x is shape {}, when should be shape {}".format(x.shape, target_shape)
        assert x.shape == target_shape, msg

        out = self.activation(x + b)
        assert len(out.shape) == 2, "Output is wrong shape!"
        #print(self.get_weights())
        return out



class TFPMultilevelDense(tf.keras.layers.Layer):

    def __init__(self, units, num_groups,
                 #multilevel_weights=False, multilevel_bias=True, fixed_prior=True,
                 make_posterior_fn, make_prior_fn,
                 kl_weight=None,
                 kl_use_exact=True,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(TFPMultilevelDense, self).__init__(**kwargs)
        self.units = units
        self.num_groups = num_groups
        #self.multilevel_weights = multilevel_weights
        #self.multilevel_bias = multilevel_bias
        #self.fixed_prior = fixed_prior
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # self.multi_w_mu = self.add_weight(
        #     shape=(self.num_groups, x_input_shape[-1], self.units),
        #     initializer='random_normal',
        #     trainable=True)
        # self.multi_b_mu = self.add_weight(
        #     shape=(self.num_groups, self.units),
        #     initializer='zeros',
        #     trainable=True)

        x_input_shape, gid_input_shape = input_shape
        #print(x_input_shape[-1])

        # Note the multiplication of the last 2 terms
        if self.multilevel_weights:
            weight_rvs = init_mean_field_vectors(
                shape=(self.num_groups, x_input_shape[-1] * self.units),
                fixed_prior=self.fixed_prior)

            if self.fixed_prior:
                self.w_mu, self.w_sigma, self.w_prior = weight_rvs
            else:
                (self.w_mu, self.w_sigma, self.w_prior,
                 self.w0_mu, self.w0_sigma, self.w0_prior) = weight_rvs
        else:
            #self.w = Dense(self.units)
            self.w = self.add_weight(
                shape=(x_input_shape[-1], self.units),
                initializer='random_normal',
                trainable=True)
            #self.w = tf.Variable(tfd.Normal(0,0.1).sample([x_input_shape[-1], self.units]))

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
                shape=(self.units,),
                initializer='zeros',
                trainable=True)
            #self.b = tf.Variable(np.zeros((self.units), dtype=np.float32))

        super(TFPMultilevelDense, self).build(input_shape)

    def construct_posterior(self, z_mu, z_sigma, z_prior, gid):
        """Build the posterior and compute the loss."""

        z_post = build_normal_variational_posterior(
            z_mu, z_sigma, gid)

        kl_loss = tfd.kl_divergence(z_post, z_prior)
        self.add_loss(tf.reduce_sum(kl_loss))
        return z_post


    def default_posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])


    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1),
                reinterpreted_batch_ndims=1)),
        ])



    def call(self, inputs):
        x, gid = inputs

        batch_size, num_features = x.shape
        assert len(x.shape) >= 2, "Data is incorrect shape!"
        assert len(gid.shape) == 1, "gid should be flat vector!"

        if self.multilevel_weights:
            w_post = self.construct_posterior(
                self.w_mu, self.w_sigma, self.w_prior, gid)

            # Sample is of shape (B, num_features*units), but
            # we want it as shape (B, num_features, units). In
            # keras you don't explicitly state batch sizes - it's
            # handled at runtime.
            w = Reshape((-1, self.units))(w_post.sample())
        else:
            # w is of shape (num_features, units)
            w = self.w

        if self.multilevel_bias:
            b_post = self.construct_posterior(
                self.b_mu, self.b_sigma, self.b_prior, gid)

            # Sample is of shape (B, units). In keras you
            # don't explicitly state batch sizes - it's
            # handled at runtime.
            b = b_post.sample()
        else:
            # w is of shape (units,)
            b = self.b

        #print('before expand', x)
        #x = tf.expand_dims(x, axis=-1)
        #print('after expand', x)
        #print(x.shape, w.shape)
        #x = tf.matmul(w, x, transpose_a=True)
        #print('after mult', x)
        #x = tf.squeeze(x, axis=-1)

        # B: batch size, p: num_features, u: num_units
        einsum_matrix_mult = '{},Bp->Bu'.format(
            'Bup' if self.multilevel_weights else 'up')
        x = tf.einsum(einsum_matrix_mult, w, x)

        #print('after squeeze', x)
        #print('bias', b)
        target_shape = (batch_size, self.units)
        msg = "x is shape {}, when should be shape {}".format(x.shape, target_shape)
        assert x.shape == target_shape, msg

        out = self.activation(x + b)
        #print('ou', out.shape)
        assert len(out.shape) == 2, "Output is wrong shape!"
        #print(out)

        print(self.get_weights())

        return out

    def get_config(self):
        config = super(TFPMultilevelDense, self).get_config()
        config.update({'units': self.units, 'num_groups': self.num_groups,
            'multilevel_weights' : False, 'multilevel_bias' : True,
            'fixed_prior' : True, 'activation' : None})
        return config



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
        #     shape=(self.num_groups, x_input_shape[-1], self.units),
        #     initializer='random_normal',
        #     trainable=True)
        # self.multi_b_mu = self.add_weight(
        #     shape=(self.num_groups, self.units),
        #     initializer='zeros',
        #     trainable=True)

        x_input_shape, gid_input_shape = input_shape
        #print(x_input_shape[-1])

        # Note the multiplication of the last 2 terms
        if self.multilevel_weights:
            weight_rvs = init_mean_field_vectors(
                shape=(self.num_groups, x_input_shape[-1] * self.units),
                fixed_prior=self.fixed_prior)

            if self.fixed_prior:
                self.w_mu, self.w_sigma, self.w_prior = weight_rvs
            else:
                (self.w_mu, self.w_sigma, self.w_prior,
                 self.w0_mu, self.w0_sigma, self.w0_prior) = weight_rvs
        else:
            #self.w = Dense(self.units)
            self.w = self.add_weight(
                shape=(x_input_shape[-1], self.units),
                initializer='random_normal',
                trainable=True)
            #self.w = tf.Variable(tfd.Normal(0,0.1).sample([x_input_shape[-1], self.units]))

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
                shape=(self.units,),
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

    def call(self, inputs):
        x, gid = inputs

        batch_size, num_features = x.shape
        assert len(x.shape) >= 2, "Data is incorrect shape!"
        assert len(gid.shape) == 1, "gid should be flat vector!"

        if self.multilevel_weights:
            w_post = self.construct_posterior(
                self.w_mu, self.w_sigma, self.w_prior, gid)

            # Sample is of shape (B, num_features*units), but
            # we want it as shape (B, num_features, units). In
            # keras you don't explicitly state batch sizes - it's
            # handled at runtime.
            w = Reshape((-1, self.units))(w_post.sample())
        else:
            # w is of shape (num_features, units)
            w = self.w

        if self.multilevel_bias:
            b_post = self.construct_posterior(
                self.b_mu, self.b_sigma, self.b_prior, gid)

            # Sample is of shape (B, units). In keras you
            # don't explicitly state batch sizes - it's
            # handled at runtime.
            b = b_post.sample()
        else:
            # w is of shape (units,)
            b = self.b

        #print('before expand', x)
        #x = tf.expand_dims(x, axis=-1)
        #print('after expand', x)
        #print(x.shape, w.shape)
        #x = tf.matmul(w, x, transpose_a=True)
        #print('after mult', x)
        #x = tf.squeeze(x, axis=-1)

        # B: batch size, p: num_features, u: num_units
        einsum_matrix_mult = '{},Bp->Bu'.format(
            'Bup' if self.multilevel_weights else 'up')
        x = tf.einsum(einsum_matrix_mult, w, x)

        #print('after squeeze', x)
        #print('bias', b)
        target_shape = (batch_size, self.units)
        msg = "x is shape {}, when should be shape {}".format(x.shape, target_shape)
        assert x.shape == target_shape, msg

        out = self.activation(x + b)
        #print('ou', out.shape)
        assert len(out.shape) == 2, "Output is wrong shape!"
        #print(out)

        print(self.get_weights())

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
        x_input_shape, gid_input_shape = input_shape

        if self.multilevel_weights:

            # Use latent dimension if provided, or else model full parameters
            if self.weights_latent_dim is not None:
                rv_dim = self.weights_latent_dim
                self.weight_factors = Dense(x_input_shape[-1] * self.units, use_bias=False)
            else:
                rv_dim = x_input_shape[-1] * self.units

            weight_rvs = init_mean_field_vectors(
                shape=(self.num_groups, rv_dim),
                fixed_prior=self.fixed_prior)
            self.reshape_weights = Reshape((x_input_shape[-1], self.units))

            if self.fixed_prior:
                self.zw_mu, self.zw_sigma, self.zw_prior = weight_rvs
            else:
                (self.zw_mu, self.zw_sigma, self.zw_prior,
                 self.zw0_mu, self.zw0_sigma, self.zw0_prior) = weight_rvs
        else:
            #self.w = Dense(self.units)
            self.w = self.add_weight(
                shape=(x_input_shape[-1], self.units),
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


    def call(self, inputs):
        x, gid = inputs

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


