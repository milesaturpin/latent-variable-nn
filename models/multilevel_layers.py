import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
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
                 multilevel_weights=False, multilevel_bias=True,
                 #fixed_prior=True,
                 make_posterior_fn=None, make_prior_fn=None,
                 kl_weight=None,
                 kl_use_exact=False,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(TFPMultilevelDense, self).__init__(**kwargs)
        self._make_posterior_fn = make_posterior_fn if make_posterior_fn is not None else self._default_posterior_trainable
        #print('NOTE: change this')
        #self._make_prior_fn = make_posterior_fn if make_posterior_fn is not None else self._default_posterior_trainable
        self._make_prior_fn = make_prior_fn if make_prior_fn is not None else self._default_prior_trainable
        self._kl_divergence_fn = _make_kl_divergence_penalty(
            kl_use_exact, weight=kl_weight)
        self.units = int(units)
        self.num_groups = num_groups
        self.multilevel_weights = multilevel_weights
        self.multilevel_bias = multilevel_bias
        #self.fixed_prior = fixed_prior
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2),
            tf.keras.layers.InputSpec(ndim=1)]


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
        last_dim = x_input_shape[-1]
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim}),
            tf.keras.layers.InputSpec(ndim=1)]

        self._posterior = self._make_posterior_fn(
            kernel_size=last_dim * self.units,
            bias_size=self.units if self.use_bias else 0)
        self._prior = self._make_prior_fn(
            last_dim * self.units,
            self.units if self.use_bias else 0)

        # # Note the multiplication of the last 2 terms
        # if self.multilevel_weights:
        #     weight_rvs = init_mean_field_vectors(
        #         shape=(self.num_groups, x_input_shape[-1] * self.units),
        #         fixed_prior=self.fixed_prior)

        #     if self.fixed_prior:
        #         self.w_mu, self.w_sigma, self.w_prior = weight_rvs
        #     else:
        #         (self.w_mu, self.w_sigma, self.w_prior,
        #          self.w0_mu, self.w0_sigma, self.w0_prior) = weight_rvs
        # else:
        #     #self.w = Dense(self.units)
        #     self.w = self.add_weight(
        #         shape=(x_input_shape[-1], self.units),
        #         initializer='random_normal',
        #         trainable=True)
        #     #self.w = tf.Variable(tfd.Normal(0,0.1).sample([x_input_shape[-1], self.units]))

        # if self.multilevel_bias:
        #     bias_rvs = init_mean_field_vectors(
        #         shape=(self.num_groups, self.units),
        #         fixed_prior=self.fixed_prior,
        #         mu_initializer='zeros')

        #     if self.fixed_prior:
        #         self.b_mu, self.b_sigma, self.b_prior = bias_rvs
        #     else:
        #         (self.b_mu, self.b_sigma, self.b_prior,
        #          self.b0_mu, self.b0_sigma, self.b0_prior) = bias_rvs
        # else:
        #     self.b = self.add_weight(
        #         shape=(self.units,),
        #         initializer='zeros',
        #         trainable=True)
        #     #self.b = tf.Variable(np.zeros((self.units), dtype=np.float32))

        super(TFPMultilevelDense, self).build(input_shape)

    def construct_posterior(self, z_mu, z_sigma, z_prior, gid):
        """Build the posterior and compute the loss."""

        z_post = build_normal_variational_posterior(
            z_mu, z_sigma, gid)

        kl_loss = tfd.kl_divergence(z_post, z_prior)
        self.add_loss(tf.reduce_sum(kl_loss))
        return z_post


    def _default_posterior_trainable(self, kernel_size, bias_size=0, dtype=None):
        # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
        print(kernel_size, bias_size)
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])


    def _default_prior_trainable(self, kernel_size, bias_size=0, dtype=None):
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

        # if self.multilevel_weights:
        #     w_post = self.construct_posterior(
        #         self.w_mu, self.w_sigma, self.w_prior, gid)

        #     # Sample is of shape (B, num_features*units), but
        #     # we want it as shape (B, num_features, units). In
        #     # keras you don't explicitly state batch sizes - it's
        #     # handled at runtime.
        #     w = Reshape((-1, self.units))(w_post.sample())
        # else:
        #     # w is of shape (num_features, units)
        #     w = self.w

        # if self.multilevel_bias:
        #     b_post = self.construct_posterior(
        #         self.b_mu, self.b_sigma, self.b_prior, gid)

        #     # Sample is of shape (B, units). In keras you
        #     # don't explicitly state batch sizes - it's
        #     # handled at runtime.
        #     b = b_post.sample()
        # else:
        #     # w is of shape (units,)
        #     b = self.b

        #print('before expand', x)
        #x = tf.expand_dims(x, axis=-1)
        #print('after expand', x)
        #print(x.shape, w.shape)
        #x = tf.matmul(w, x, transpose_a=True)
        #print('after mult', x)
        #x = tf.squeeze(x, axis=-1)

        q = self._posterior(x)
        r = self._prior(x)
        #print(q,r)
        self.add_loss(self._kl_divergence_fn(q, r))
        #kl_loss = tfd.kl_divergence(q, r)
        #self.add_loss(tf.reduce_sum(kl_loss))

        w = tf.convert_to_tensor(value=q)
        prev_units = self.input_spec[0].axes[-1]
        if self.use_bias:
          split_sizes = [prev_units * self.units, self.units]
          kernel, bias = tf.split(w, split_sizes, axis=-1)
        else:
          kernel, bias = w, None

        kernel = tf.reshape(kernel, shape=tf.concat([
            tf.shape(input=kernel)[:-1],
            [prev_units, self.units],
        ], axis=0))

        #outputs = tf.matmul(inputs, kernel)


        # B: batch size, p: num_features, u: num_units
        einsum_matrix_mult = '{},Bp->Bu'.format(
            'Bup' if self.multilevel_weights else 'up')
        #x = tf.einsum(einsum_matrix_mult, w, x)
        outputs = tf.einsum(einsum_matrix_mult, kernel, x)


        #print('after squeeze', x)
        #print('bias', b)
        target_shape = (batch_size, self.units)
        msg = "x is shape {}, when should be shape {}".format(x.shape, target_shape)
        assert x.shape == target_shape, msg

        #out = self.activation(x + b)
        #print('ou', out.shape)
        assert len(outputs.shape) == 2, "Output is wrong shape!"
        #print(out)

        #print(self.get_weights())

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def get_config(self):
        config = super(TFPMultilevelDense, self).get_config()
        # TODO: actually fill in
        config.update({'units': self.units, 'num_groups': self.num_groups,
            'multilevel_weights' : False, 'multilevel_bias' : True,
            'fixed_prior' : True, 'activation' : None})
        return config




def _make_kl_divergence_penalty(
    use_exact_kl=False,
    test_points_reduce_axis=(),  # `None` == "all"; () == "none".
    test_points_fn=tf.convert_to_tensor,
    weight=None):
    """Creates a callable computing `KL[a,b]` from `a`, a `tfd.Distribution`."""

    if use_exact_kl:
        kl_divergence_fn = kullback_leibler.kl_divergence
    else:
        def kl_divergence_fn(distribution_a, distribution_b):
            z = test_points_fn(distribution_a)
            return tf.reduce_mean(
                input_tensor=distribution_a.log_prob(z) - distribution_b.log_prob(z),
                axis=test_points_reduce_axis)

    # Closure over: kl_divergence_fn, weight.
    def _fn(distribution_a, distribution_b):
        """Closure that computes KLDiv as a function of `a` as in `KL[a, b]`."""
        with tf1.name_scope('kldivergence_loss'):
            kl = kl_divergence_fn(distribution_a, distribution_b)
            if weight is not None:
                kl = tf.cast(weight, dtype=kl.dtype) * kl
            # Losses appended with the model.add_loss and are expected to be a single
            # scalar, unlike model.loss, which is expected to be the loss per sample.
            # Therefore, we reduce over all dimensions, regardless of the shape.
            # We take the sum because (apparently) Keras will add this to the *post*
            # `reduce_sum` (total) loss.
            # TODO(b/126259176): Add end-to-end Keras/TFP test to ensure the API's
            # align, particularly wrt how losses are aggregated (across batch
            # members).
            return tf.reduce_sum(input_tensor=kl,
                                 name='batch_total_kl_divergence')

    return _fn





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


