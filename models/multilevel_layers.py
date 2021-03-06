import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, Reshape, Flatten)
#from tensorflow_probability.layers import DenseVariational

from models.base_model import BaseModel
from models.multilevel_variables import HierarchicalVariableLayer
from models.model_utils import (
    init_mean_field_vectors, build_normal_variational_posterior,
    latent_normal_matrix, latent_matrix_variational_posterior, softplus_inverse)

from tensorflow.keras.utils import to_categorical

import ipdb

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



class MyHierarchicalDense(tf.keras.layers.Layer):

    def __init__(self, units, num_groups,
                 multilevel_weights=True, 
                 multilevel_bias=True,
                 kl_weight=1.,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(MyHierarchicalDense, self).__init__(**kwargs)
        self.units = int(units)
        self.num_groups = num_groups
        self.multilevel_weights = multilevel_weights
        self.multilevel_bias = multilevel_bias
        self.kl_weight = kl_weight
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        #self.is_hierarchical = True
#         self.input_spec = [
#             tf.keras.layers.InputSpec(min_ndim=2),
#             tf.keras.layers.InputSpec(ndim=1)]
        
    def build(self, input_shape):
        x_input_shape, gid_input_shape = input_shape
        last_dim = x_input_shape[-1]
#         self.input_spec = [
#             tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim}),
#             tf.keras.layers.InputSpec(ndim=1)]
        
        self.w = HierarchicalVariableLayer(
            variable_shape=[self.units, last_dim],
            num_groups=self.num_groups, 
            kl_weight=self.kl_weight,
            name=self.name+'/kernel'
            )
        
        #if self.multilevel_bias:
        self.b = HierarchicalVariableLayer(
            variable_shape=[self.units],
            num_groups=self.num_groups, 
            kl_weight=self.kl_weight,
            name=self.name+'/bias'
            )
        
        super(MyHierarchicalDense, self).build(input_shape)
     
    @tf.function
    def call(self, inputs):
        
        x, gid = inputs
        batch_size, num_features = x.shape
        # Sanity checks
        #assert len(x.shape) >= 2, "Data is incorrect shape!"
        #assert len(gid.shape) == 1, "gid should be flat vector!"
        
        w = self.w(gid)
        b = self.b(gid)
            
        # B: batch size, p: num_features, u: num_units
        einsum_matrix_mult = '{},Bp->Bu'.format(
            'Bup' if self.multilevel_weights else 'up')
        outputs = tf.einsum(einsum_matrix_mult, w, x)

        # Sanity checks
        target_shape = (batch_size, self.units)
        msg = "output is shape {}, when should be shape {}".format(outputs.shape, target_shape)
        assert outputs.shape == target_shape, msg
        assert len(outputs.shape) == 2, "Output is wrong shape!"

        if self.use_bias:
            #outputs = tf.nn.bias_add(outputs, b)
            outputs = outputs + b

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class MyMultilevelDense(tf.keras.layers.Layer):
    """

    """

    def __init__(self, units, num_groups,
                 multilevel_weights=True, multilevel_bias=True,
                 #fixed_prior=True,
                 kl_weight=None,
                 kl_use_exact=False,
                 group_kl_weights=1.,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(MyMultilevelDense, self).__init__(**kwargs)
        #self._kl_divergence_fn = _make_kl_divergence_penalty(
        #    kl_use_exact, weight=kl_weight)
        self.units = int(units)
        self.num_groups = num_groups
        self.multilevel_weights = multilevel_weights
        self.multilevel_bias = multilevel_bias
        self.group_kl_weights = group_kl_weights
        #self.fixed_prior = fixed_prior
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2),
            tf.keras.layers.InputSpec(ndim=1)]


    def build(self, input_shape):

        x_input_shape, gid_input_shape = input_shape
        last_dim = x_input_shape[-1]
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim}),
            tf.keras.layers.InputSpec(ndim=1)]

        # Initialization schemes
        # Set priors on variances to be -5 since softplus(-5+c)=0.0008

        # Variational parameters for z_k, the group latent variables
        # Mean of the variational posterior for group latents
        self.w_mu_k = self.add_weight(
            shape=(self.num_groups, self.units, last_dim),
            initializer='random_normal',
            trainable=True,
            name='w_mu_k')
        # Variance of the variational posterior for group latents
        self.w_sigma_k = self.add_weight(
            shape=(self.num_groups, self.units, last_dim),
            initializer=tf.constant_initializer(-4.),
            #initializer=tf.constant_initializer(1e-4),
            trainable=True,
            name='w_sigma_k')

        # Variational parameters for z_0, the mean of the hierarchical prior
        # Mean of the variational posterior for group prior z_0 
        self.w_mu0 = self.add_weight(
            shape=(self.units, last_dim),
            initializer='zeros',
            trainable=True,
            name='w_mu0')
        # Variance of the variational posterior for the group mean z_0
        self.w_sigma0 = self.add_weight(
            shape=(self.units, last_dim),
            # Should be relatively diffuse, softplus(0+c)=1
            initializer=tf.constant_initializer(-4.),
            #initializer=tf.constant_initializer(1e-4),
            #tf.constant_initializer(100.)
            trainable=True,
            name='w_sigma0')

        # TODO: create variational params for tau_k
        # Variational parameters for tau_k, the group variance of the hierarchical prior
        # Mean of the variational posterior for group prior variance tau_k 
        self.w_tau_k_mu = self.add_weight(
            shape=(self.num_groups,),
            initializer=tf.constant_initializer(-1),
            trainable=True,
            name='w_tau_k_mu')
        # Variance of the variational posterior for the group prior variance tau_k
        self.w_tau_k_sigma = self.add_weight(
            shape=(self.num_groups,),
            # Should be relatively diffuse, softplus(0+c)=1
            initializer=tf.constant_initializer(-2),
            #tf.constant_initializer(100.)
            trainable=True,
            name='w_tau_k_sigma')

        # # Fixed parameters for hyperprior over z_0 ~ N(0, v^-1)
        # # Fixed mean of 0
        # self.w_z0_prior_mean = self.add_weight(
        #     shape=(self.units, last_dim),
        #     initializer=tf.constant_initializer(0.),
        #     trainable=False,
        #     name='w_z0_prior_mean')
        # # Fixed variance
        # self.w_z0_prior_variance = self.add_weight(
        #     shape=(self.units, last_dim),
        #     initializer=tf.constant_initializer(10.),
        #     trainable=False,
        #     name='w_z0_prior_mean')

        # # Fixed parameters for hyperprior over tau_k ~ N(0, tau_0)
        # # Note that this value is unbounded because we can square later
        # # Fixed mean of 0
        # self.w_tau_k_prior_mean = self.add_weight(
        #     shape=(self.num_groups,),
        #     initializer=tf.constant_initializer(0.),
        #     trainable=False,
        #     name='w_tau_k_prior_mean')
        # # Fixed hyperprior over tau_k, aka tau_0
        # self.w_tau_k_prior_variance = self.add_weight(
        #     shape=(self.num_groups,),
        #     initializer=tf.constant_initializer(10.),
        #     trainable=False,
        #     name='w_tau_k_prior_variance')
                # Fixed parameters for hyperprior over z_0 ~ N(0, v^-1)
        # Fixed mean of 0
        self.w_z0_prior_mean = 0.
        # Fixed variance
        self.w_z0_prior_variance = 100

        # Fixed parameters for hyperprior over tau_k ~ N(0, tau_0)
        # Note that this value is unbounded because we can square later
        # Fixed mean of 0
        self.w_tau_k_prior_mean = 0.
        # Fixed hyperprior over tau_k, aka tau_0
        self.w_tau_k_prior_variance = 100.




                # Variational parameters for z_k, the group latent variables
        # Mean of the variational posterior for group latents
        self.b_mu_k = self.add_weight(
            shape=(self.num_groups, self.units),
            initializer='zeros',
            trainable=True,
            name='b_mu_k')
        # Variance of the variational posterior for group latents
        self.b_sigma_k = self.add_weight(
            shape=(self.num_groups, self.units),
            initializer=tf.constant_initializer(-4),
            #initializer=tf.constant_initializer(1e-4),
            trainable=True,
            name='b_sigma_k')

        # Variational parameters for z_0, the mean of the hierarchical prior
        # Mean of the variational posterior for group prior z_0 
        self.b_mu0 = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
            name='b_mu0')
        # Variance of the variational posterior for the group mean z_0
        self.b_sigma0 = self.add_weight(
            shape=(self.units,),
            # Should be relatively diffuse, softplus(0+c)=1
            initializer=tf.constant_initializer(-4.),
            #initializer=tf.constant_initializer(1e-4),
            trainable=True,
            name='b_sigma0')

        # Variational parameters for tau_k, the group variance of the hierarchical prior
        # Mean of the variational posterior for group prior variance tau_k 
        self.b_tau_k_mu = self.add_weight(
            shape=(self.num_groups,),
            initializer=tf.constant_initializer(-1),
            trainable=True,
            name='b_tau_k_mu')
        # Variance of the variational posterior for the group prior variance tau_k
        self.b_tau_k_sigma = self.add_weight(
            shape=(self.num_groups,),
            # Should be relatively diffuse, softplus(0+c)=1
            initializer=tf.constant_initializer(-2),
            #tf.constant_initializer(100.)
            trainable=True,
            name='b_tau_k_sigma')

        self.b_z0_prior_mean = 0.
        # Fixed variance
        self.b_z0_prior_variance = 100

        # Fixed parameters for hyperprior over tau_k ~ N(0, tau_0)
        # Note that this value is unbounded because we can square later
        # Fixed mean of 0
        self.b_tau_k_prior_mean = 0.
        # Fixed hyperprior over tau_k, aka tau_0
        self.b_tau_k_prior_variance = 1e5


        super(MyMultilevelDense, self).build(input_shape)

    @tf.function
    def sample_posterior(self, mu, sigma):
        # By sampling after gather, I use different noise for each sample
        eps = np.random.randn(*mu.shape)
        samp = mu + sigma*eps
        return samp

    @tf.function
    def compute_kl(self, mu1, sigma1, mu2, sigma2):

        # # Hack to fix broadcasting failure e.g. divide [5]/[5,62,128]
        # dim_diff = len(sigma1.shape) - len(sigma2.shape)
        # for _ in range(dim_diff):
        #     sigma2 = tf.expand_dims(sigma2, axis=[-1])

        kl = (
            tf.math.log(sigma2/sigma1)
            + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2)
            - 0.5)

        return kl


    
    def call(self, inputs):
        x, gid = inputs

        #import ipdb; ipdb.set_trace()

        batch_size, num_features = x.shape
        # Sanity checks
        assert len(x.shape) >= 2, "Data is incorrect shape!"
        assert len(gid.shape) == 1, "gid should be flat vector!"

        # Get a softplussed version of the variances
        c = np.log(np.expm1(1.))
        var_activation = tf.math.exp
        w_sigma_k_sftpls = var_activation(c + self.w_sigma_k)
        b_sigma_k_sftpls = var_activation(c + self.b_sigma_k)
        w_sigma0_sftpls = var_activation(c + self.w_sigma0)
        b_sigma0_sftpls = var_activation(c + self.b_sigma0)
        w_tau_k_sigma_sftpls = var_activation(c + self.w_tau_k_sigma)
        b_tau_k_sigma_sftpls = var_activation(c + self.b_tau_k_sigma)

        gather = lambda x: tf.gather(x, gid)

        w = self.sample_posterior(gather(self.w_mu_k), gather(w_sigma_k_sftpls))
        b = self.sample_posterior(gather(self.b_mu_k), gather(b_sigma_k_sftpls))

        w0 = self.sample_posterior(self.w_mu0, w_sigma0_sftpls)
        b0 = self.sample_posterior(self.b_mu0, b_sigma0_sftpls)

        w_tau_k = self.sample_posterior(gather(self.w_tau_k_mu), gather(w_tau_k_sigma_sftpls))
        b_tau_k = self.sample_posterior(gather(self.b_tau_k_mu), gather(b_tau_k_sigma_sftpls))
        
        w_tau_k_sq = w_tau_k ** 2
        b_tau_k_sq = b_tau_k ** 2
        
        # KL between var post on z0 and fixed prior over z0
        w_z0_kl_loss = self.compute_kl(
            self.w_mu0, w_sigma0_sftpls, self.w_z0_prior_mean, self.w_z0_prior_variance)
        b_z0_kl_loss = self.compute_kl(
            self.b_mu0, b_sigma0_sftpls, self.b_z0_prior_mean, self.b_z0_prior_variance)


        # KL between var post on tau_k and fixed prior over tau_k
        w_tau_k_kl_loss = self.compute_kl(
            gather(self.w_tau_k_mu), gather(w_tau_k_sigma_sftpls),
            self.w_tau_k_prior_mean, self.w_tau_k_prior_variance)
        b_tau_k_kl_loss = self.compute_kl(
            gather(self.b_tau_k_mu), gather(b_tau_k_sigma_sftpls), 
            self.b_tau_k_prior_mean, self.b_tau_k_prior_variance)


        # TODO: pick version
        # Version 1: taking the KL between q(z_k) and p(z_k), have to sample 
        # the parameters of p(z_k|z_0, tau_k) from q(z_0, tau_k)
        w_z_k_kl_loss = self.compute_kl(gather(self.w_mu_k), gather(w_sigma_k_sftpls), w0, 
            tf.expand_dims(tf.expand_dims(w_tau_k_sq, axis=[-1]), axis=[-1])
            )
        b_z_k_kl_loss = self.compute_kl(gather(self.b_mu_k), gather(b_sigma_k_sftpls), b0, 
            tf.expand_dims(b_tau_k_sq, axis=[-1]))

        # Version 2: take the expected log prob of z_k wrt q.
        #   Version 2a: do just a monte carlo estimate
        w_z_k_log_prob = tfd.Normal(w0, 
            tf.expand_dims(tf.expand_dims(w_tau_k_sq, axis=[-1]), axis=[-1])
            ).log_prob(w)
        b_z_k_log_prob = tfd.Normal(b0, 
            tf.expand_dims(b_tau_k_sq, axis=[-1])
            ).log_prob(b)
        #   Version 2b: do mix of closed form and monte carlo
        #import ipdb; ipdb.set_trace()
        D = self.w_mu_k.shape[1] * self.w_mu_k.shape[2]
        w_z_k_log_prob = tf.squeeze(
            -1/2 * D * np.log(2*np.pi) 
            - 1/2 * D * tf.math.log(w_tau_k_sq)
            - 1/2/w_tau_k_sq * (
                tf.reduce_sum(gather(w_sigma_k_sftpls), axis=[-1,-2]) 
                + tf.reduce_sum(w_sigma0_sftpls, axis=[-1,-2]) 
                + tf.einsum('Bij, Bij -> ', gather(self.w_mu_k), gather(self.w_mu_k))
                + tf.einsum('ij, ij -> ', self.w_mu0, self.w_mu0)
                - 2 * tf.einsum('Bij, ij -> ', gather(self.w_mu_k), 
                    self.w_mu0)))

        D = self.b_mu_k.shape[1]
        b_z_k_log_prob = tf.squeeze(
            -1/2 * D * np.log(2*np.pi) 
            - 1/2 * D * tf.math.log(b_tau_k_sq)
            - 1/2/b_tau_k_sq * (
                tf.reduce_sum(gather(b_sigma_k_sftpls), axis=[-1]) 
                + tf.reduce_sum(b_sigma0_sftpls, axis=[-1]) 
                + tf.einsum('Bi, Bi -> ', gather(self.b_mu_k), gather(self.b_mu_k))
                + tf.einsum('i, i -> ', self.b_mu0, self.b_mu0)
                - 2 * tf.einsum('Bi, i -> ', gather(self.b_mu_k), self.b_mu0))) 

        #import ipdb; ipdb.set_trace()

        # Scale kl losses
        # VERSION 1

        #import ipdb; ipdb.set_trace()

        group_kl_weights = gather(self.group_kl_weights)
        group_kl_weights1 = tf.expand_dims(tf.expand_dims(group_kl_weights, axis=[-1]), axis=[-1])
        group_kl_weights2 = tf.expand_dims(group_kl_weights, axis=[-1])
        # Take inverse then sum to get train size
        train_size = tf.reduce_sum(1. / self.group_kl_weights)
        prior_kl_weight = 1. / train_size

        #w_z_k_log_prob = tf.math.multiply(b_z_k_log_prob,tf.expand_dims(tf.expand_dims(group_kl_weights, axis=[-1]), axis=[-1]))

        w_z0_kl_loss = tf.math.multiply(w_z0_kl_loss, prior_kl_weight)
        b_z0_kl_loss = tf.math.multiply(b_z0_kl_loss, prior_kl_weight)
        # adding squeeze for lin reg case

        from string import ascii_letters

        # e.g. get first three letters of alphabet
        first_shape = ascii_letters[:len(tf.squeeze(w_z_k_log_prob).shape)]
        w_z_k_log_prob = tf.einsum('{}, a ->{}'.format(first_shape, first_shape), 
            tf.squeeze(w_z_k_log_prob), group_kl_weights)
        first_shape = ascii_letters[:len(tf.squeeze(b_z_k_log_prob).shape)]
        b_z_k_log_prob = tf.einsum('{}, a ->{}'.format(first_shape, first_shape), 
            tf.squeeze(b_z_k_log_prob), group_kl_weights)
        #w_z_k_log_prob = tf.math.multiply(tf.squeeze(w_z_k_log_prob), group_kl_weights1)
        #b_z_k_log_prob = tf.math.multiply(tf.squeeze(b_z_k_log_prob), group_kl_weights2)
        # Weird broadcasting thing happening so make these group_kl_weights 
        # instead of group_kl_weights1
        w_tau_k_kl_loss = tf.math.multiply(w_tau_k_kl_loss, group_kl_weights)
        b_tau_k_kl_loss = tf.math.multiply(b_tau_k_kl_loss, group_kl_weights)

        # scale kl VERSION 2 TODO: I actually need to change the above KL computations too,
        # like for z_k and tau_k I won't actually do gather
        # scale_kl = lambda x: tf.math.multiply(x, prior_kl_weight)
        # w_z0_kl_loss = scale_kl(w_z0_kl_loss)
        # b_z0_kl_loss = scale_kl(b_z0_kl_loss)
        # w_z_k_log_prob = scale_kl(w_z_k_log_prob)
        # b_z_k_log_prob = scale_kl(b_z_k_log_prob)
        # w_tau_k_kl_loss = scale_kl(w_tau_k_kl_loss)
        # b_tau_k_kl_loss = scale_kl(b_tau_k_kl_loss)


        # TODO: scale KL VERSION 3, interpolate bt 1 and 2

        

        kl1 = tf.reduce_sum(w_z0_kl_loss)
        kl2 = tf.reduce_sum(w_tau_k_kl_loss)
        kl3 = tf.reduce_sum(b_z0_kl_loss)
        kl4 = tf.reduce_sum(b_tau_k_kl_loss) 
        kl5 = tf.reduce_sum(w_z_k_log_prob)
        kl6 = tf.reduce_sum(b_z_k_log_prob)
        
        #import ipdb; ipdb.set_trace()
        # don't add these, these are the other versions
        w_z_k_kl_loss = tf.math.multiply(w_z_k_kl_loss, group_kl_weights1)
        b_z_k_kl_loss = tf.math.multiply(b_z_k_kl_loss, group_kl_weights2)
        kl7 = tf.reduce_sum(w_z_k_kl_loss)
        kl8 = tf.reduce_sum(b_z_k_kl_loss)

        kls = [kl1, kl2, kl3, kl4, kl5, kl6, kl7, kl8]

        kl_losses = (kl1 + kl2 + kl3 + kl4 - kl5 - kl6)

        #self.add_loss(kl1)
        #self.add_loss(kl2)
        #self.add_loss(kl3)
        #self.add_loss(kl4)
        #self.add_loss(-1*kl5)
        #self.add_loss(-1*kl6)
        self.add_loss(kl7)
        self.add_loss(kl8)

        # MAKE SURE TO TRN THIS BACK ON!!!!!!!
        # MAKE SURE TO TRN THIS BACK ON!!!!!!!
        # MAKE SURE TO TRN THIS BACK ON!!!!!!!
        # MAKE SURE TO TRN THIS BACK ON!!!!!!!
        # MAKE SURE TO TRN THIS BACK ON!!!!!!!
        #self.add_loss(kls)
        #self.add_loss(kl_losses)

        # TODO: sample w0_mu etc. and pass into compute KL, then compute KL of w0_mu with its prior
        # Well I guess im still not certain about whether it is the KL or the log prob,
        # I guess I'll just try both. But yeah sample z_0 and tau_k from q(w0_mu, w0_sigma) and 
        # q(tau_k_mu, tau_k_sigma)

        #w_kl_loss = self.compute_kl(self.w_mu, self.w_sigma, self.w0_mu, self.w0_sigma, gid)
        #b_kl_loss = self.compute_kl(self.b_mu, self.b_sigma, self.b0_mu, self.b0_sigma, gid)
        #self.add_loss(tf.reduce_sum(w_kl_loss) + tf.reduce_sum(b_kl_loss))

        

        # B: batch size, p: num_features, u: num_units
        einsum_matrix_mult = '{},Bp->Bu'.format(
            'Bup' if self.multilevel_weights else 'up')
        #x = tf.einsum(einsum_matrix_mult, w, x)
        #print(w.shape, x.shape)
        outputs = tf.einsum(einsum_matrix_mult, w, x)

        # Sanity checks
        target_shape = (batch_size, self.units)
        msg = "output is shape {}, when should be shape {}".format(outputs.shape, target_shape)
        assert outputs.shape == target_shape, msg
        assert len(outputs.shape) == 2, "Output is wrong shape!"

        if self.use_bias:
            #outputs = tf.nn.bias_add(outputs, b)
            outputs = outputs + b

        if self.activation is not None:
            outputs = self.activation(outputs)

        #.set_trace()

        return outputs

    def get_config(self):
        config = super(MyMultilevelDense, self).get_config()
        config.update({
            'units': self.units,
            'num_groups': self.num_groups,
            'multilevel_weights' : self.multilevel_weights,
            'multilevel_bias' : self.multilevel_bias,
            'use_bias' : self.use_bias,
            'activation' : self.activation,
            'input_spec' : self.input_spec})
        return config



class MAPMultilevelDense(tf.keras.layers.Layer):

    def __init__(self, units, num_groups,
                 multilevel_weights=True, multilevel_bias=True,
                 activation=None,
                 use_bias=True,
                 regularization_strength=1,
                 **kwargs):
        super(MAPMultilevelDense, self).__init__(**kwargs)
        #self._kl_divergence_fn = _make_kl_divergence_penalty(
        #    kl_use_exact, weight=kl_weight)
        self.units = int(units)
        self.num_groups = num_groups
        self.multilevel_weights = multilevel_weights
        self.multilevel_bias = multilevel_bias
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.regularization_strength = regularization_strength
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2),
            tf.keras.layers.InputSpec(ndim=1)]


    def build(self, input_shape):

        x_input_shape, gid_input_shape = input_shape
        last_dim = x_input_shape[-1]
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim}),
            tf.keras.layers.InputSpec(ndim=1)]

        # self.w_mu = self.add_weight(
        #     shape=(self.num_groups, self.units, last_dim),
        #     initializer='random_normal',
        #     trainable=True,
        #     name='kernel_mu')

        # Wanted to use glorot normal but depends on shape so initialized first and then grouped together
        # Had to use initializer instead of self.add_weight
        create_weight_matrix = lambda : (
            tf.initializers.glorot_normal()(shape=(self.units, last_dim)))
        weight_matrix_list = [create_weight_matrix() for _ in range(self.num_groups)]

        self.w_mu = tf.Variable(
            np.stack(weight_matrix_list),
            name='kernel_mu')


        self.b_mu = self.add_weight(
            shape=(self.num_groups, self.units),
            initializer='random_normal',
            trainable=True,
            name='bias_mu')
        # Adding this because was being weird when both this and prior 0
        self.b_mu + 0.0001

        self.w0_mu = self.add_weight(
            shape=(self.units, last_dim),
            initializer='glorot_normal',
            trainable=True,
            name='kernel_mu_prior')
        self.w0_mu = tf.expand_dims(self.w0_mu, axis=0)

        self.b0_mu = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias_mu_prior')
        self.b0_mu = tf.expand_dims(self.b0_mu, axis=0)

        super(MAPMultilevelDense, self).build(input_shape)

    def call(self, inputs):
        x, gid = inputs

        batch_size, num_features = x.shape
        assert len(x.shape) >= 2, "Data is incorrect shape!"
        assert len(gid.shape) == 1, "gid should be flat vector!"

        w = tf.gather(self.w_mu, gid)
        b = tf.gather(self.b_mu, gid)

        # B: batch size, p: num_features, u: num_units
        einsum_matrix_mult = '{},Bp->Bu'.format(
            'Bup' if self.multilevel_weights else 'up')
        #x = tf.einsum(einsum_matrix_mult, w, x)
        #print(w.shape, x.shape)
        outputs = tf.einsum(einsum_matrix_mult, w, x)

        target_shape = (batch_size, self.units)
        msg = "output is shape {}, when should be shape {}".format(outputs.shape, target_shape)
        assert outputs.shape == target_shape, msg
        assert len(outputs.shape) == 2, "Output is wrong shape!"

        if self.use_bias:
            #outputs = tf.nn.bias_add(outputs, b)
            outputs = outputs + b

        if self.activation is not None:
            outputs = self.activation(outputs)

        # Regularize with "prior"

        l2_prior_cost = lambda x, y, axes: (
            tf.reduce_sum(tf.math.square(x - y), axis=axes))
        w_loss = l2_prior_cost(w, self.w0_mu, axes=[1, 2])
        b_loss = l2_prior_cost(b, self.b0_mu, axes=[1])
        #print(w_loss, b_loss)
        #print((tf.reduce_sum(w_loss) + tf.reduce_sum(b_loss)))
        #print(self.regularization_strength
        #    * (tf.reduce_sum(w_loss) + tf.reduce_sum(b_loss)))

        self.add_loss(
            self.regularization_strength
            * (tf.reduce_sum(w_loss) + tf.reduce_sum(b_loss)))

        return outputs

    def get_config(self):
        config = super(MAPMultilevelDense, self).get_config()
        config.update({
            'units': self.units,
            'num_groups': self.num_groups,
            'multilevel_weights' : self.multilevel_weights,
            'multilevel_bias' : self.multilevel_bias,
            'use_bias' : self.use_bias,
            'activation' : self.activation,
            'input_spec' : self.input_spec})
        return config



class MAPFactoredMultilevelDense(tf.keras.layers.Layer):

    def __init__(self, units, num_groups,
                 multilevel_weights=True, multilevel_bias=True,
                 weights_latent_dim=None, bias_latent_dim=None,
                 activation=None,
                 use_bias=True,
                 regularization_strength=1,
                 **kwargs):
        super(MAPFactoredMultilevelDense, self).__init__(**kwargs)
        #self._kl_divergence_fn = _make_kl_divergence_penalty(
        #    kl_use_exact, weight=kl_weight)
        self.units = int(units)
        self.num_groups = num_groups
        self.multilevel_weights = multilevel_weights
        self.multilevel_bias = multilevel_bias
        self.use_bias = use_bias
        self.weights_latent_dim = weights_latent_dim
        self.bias_latent_dim = bias_latent_dim
        self.activation = tf.keras.activations.get(activation)
        self.regularization_strength = regularization_strength
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2),
            tf.keras.layers.InputSpec(ndim=1)]


    def build(self, input_shape):

        x_input_shape, gid_input_shape = input_shape
        last_dim = x_input_shape[-1]
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim}),
            tf.keras.layers.InputSpec(ndim=1)]



        if self.multilevel_weights:
            # Use latent dimension if provided, or else model full parameters
            if self.weights_latent_dim is not None:
                rv_dim = self.weights_latent_dim
                self.weight_factors = Dense(last_dim * self.units, use_bias=False)
            else:
                rv_dim = last_dim * self.units

            self.z_kernel = self.add_weight(
                shape=(self.num_groups, rv_dim),
                initializer='random_normal',
                trainable=True,
                name='z_kernel')
            self.reshape_weights = Reshape((self.units, last_dim))
        else:
            #self.w = Dense(self.units)
            self.w = self.add_weight(
                shape=(self.units, last_dim),
                initializer='random_normal',
                trainable=True)

        if self.multilevel_bias:
            # Use latent dimension if provided, or else model full parameters
            if self.bias_latent_dim is not None:
                rv_dim = self.bias_latent_dim
                self.bias_factors = Dense(self.units, use_bias=False)
            else:
                rv_dim = self.units

            self.z_bias = self.add_weight(
                shape=(self.num_groups, rv_dim),
                initializer='random_normal',
                trainable=True,
                name='z_bias')

        else:
            self.b = self.add_weight(
                shape=(self.units),
                initializer='zeros',
                trainable=True)

    def call(self, inputs):
        x, gid = inputs

        batch_size, num_features = x.shape
        assert len(x.shape) >= 2, "Data is incorrect shape!"
        assert len(gid.shape) == 1, "gid should be flat vector!"

        z_kernel = tf.gather(self.z_kernel, gid)
        w = self.weight_factors(z_kernel)
        w = self.reshape_weights(w)

        z_bias = tf.gather(self.z_bias, gid)
        b = self.bias_factors(z_bias)

        # B: batch size, p: num_features, u: num_units
        einsum_matrix_mult = '{},Bp->Bu'.format(
            'Bup' if self.multilevel_weights else 'up')
        #x = tf.einsum(einsum_matrix_mult, w, x)
        #print(w.shape, x.shape)
        outputs = tf.einsum(einsum_matrix_mult, w, x)

        target_shape = (batch_size, self.units)
        msg = "output is shape {}, when should be shape {}".format(outputs.shape, target_shape)
        assert outputs.shape == target_shape, msg
        assert len(outputs.shape) == 2, "Output is wrong shape!"

        if self.use_bias:
            #outputs = tf.nn.bias_add(outputs, b)
            outputs = outputs + b

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def get_config(self):
        config = super(MAPFactoredMultilevelDense, self).get_config()
        config.update({
            'units': self.units,
            'num_groups': self.num_groups,
            'multilevel_weights' : self.multilevel_weights,
            'multilevel_bias' : self.multilevel_bias,
            'use_bias' : self.use_bias,
            'activation' : self.activation,
            'input_spec' : self.input_spec})
        return config


class TFPDense(tf.keras.layers.Layer):

    def __init__(self, units, num_groups,
                 multilevel_weights=False, multilevel_bias=True,
                 #fixed_prior=True,
                 make_posterior_fn=None, make_prior_fn=None,
                 kl_weight=None,
                 kl_use_exact=False,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(TFPDense, self).__init__(**kwargs)
        self._make_posterior_fn = make_posterior_fn if make_posterior_fn is not None else self._default_posterior_trainable
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

        super(TFPDense, self).build(input_shape)

    def construct_posterior(self, z_mu, z_sigma, z_prior, gid):
        """Build the posterior and compute the loss."""

        z_post = build_normal_variational_posterior(
            z_mu, z_sigma, gid)

        kl_loss = tfd.kl_divergence(z_post, z_prior)
        self.add_loss(tf.reduce_sum(kl_loss))
        return z_post


    def _default_posterior_trainable(self, kernel_size, bias_size=0, dtype=None):
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

        q = self._posterior(x)
        r = self._prior(x)
        #print(q,r)
        #self.add_loss(self._kl_divergence_fn(q, r))
        kl_loss = tfd.kl_divergence(q, r)
        self.add_loss(tf.reduce_sum(kl_loss))

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

        target_shape = (batch_size, self.units)
        msg = "x is shape {}, when should be shape {}".format(x.shape, target_shape)
        assert x.shape == target_shape, msg
        assert len(outputs.shape) == 2, "Output is wrong shape!"

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs




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

        x_input_shape, gid_input_shape = input_shape
        last_dim = x_input_shape[-1]
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim}),
            tf.keras.layers.InputSpec(ndim=1)]

        self._posterior = self._make_posterior_fn(
            num_groups=self.num_groups,
            kernel_size=last_dim * self.units,
            bias_size=self.units if self.use_bias else 0)
        self._prior = self._make_prior_fn(
            last_dim * self.units,
            self.units if self.use_bias else 0)

        super(TFPMultilevelDense, self).build(input_shape)

    def construct_posterior(self, z_mu, z_sigma, z_prior, gid):
        """Build the posterior and compute the loss."""

        z_post = build_normal_variational_posterior(
            z_mu, z_sigma, gid)

        kl_loss = tfd.kl_divergence(z_post, z_prior)
        self.add_loss(tf.reduce_sum(kl_loss))
        return z_post


    def _default_posterior_trainable(self, num_groups, kernel_size, bias_size, dtype=None):
        # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer([num_groups, 2 * n], dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1
                )),
        ])
        # gid = tf.keras.layers.Input(shape=(1,))
        # params = tfp.layers.VariableLayer([num_groups, 2 * n], dtype=dtype)
        # batch_params = tf.gather(params, gid)
        # batch_dist = tfp.layers.DistributionLambda(lambda t: tfd.Independent(
        #     tfd.Normal(loc=t[..., :n],
        #                 scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
        #     reinterpreted_batch_ndims=1))(batch_params)
        # posterior = tf.keras.models.Model(inputs=gid, outputs=batch_dist)
        # return posterior


    def _default_prior_trainable(self, kernel_size, bias_size, dtype=None):
        # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])


    def call(self, inputs):
        x, gid = inputs

        batch_size, num_features = x.shape
        assert len(x.shape) >= 2, "Data is incorrect shape!"
        assert len(gid.shape) == 1, "gid should be flat vector!"

        q = self._posterior(x)

        r = self._prior(x)

        print( q,r)
        #print(self._kl_divergence_fn(q, r))
        self.add_loss(self._kl_divergence_fn(q, r))
        print('before', q)
        q = tf.gather(q, gid)
        print('after', q)
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

        target_shape = (batch_size, self.units)
        msg = "x is shape {}, when should be shape {}".format(x.shape, target_shape)
        assert x.shape == target_shape, msg
        assert len(outputs.shape) == 2, "Output is wrong shape!"

        if self.use_bias:
            # outputs = tf.nn.bias_add(outputs, bias)
            outputs = outputs + bias

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs




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


