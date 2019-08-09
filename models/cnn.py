import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, Reshape, Flatten)

from models.base_model import BaseModel
from models.model_utils import (
    latent_normal_vector, latent_vector_variational_posterior)

tfd = tfp.distributions

"""
# TODO: possibly merge with normal model call
def call_sample_z(self, x, gid, num_samples=1, importance_weighted=True):

    #Forward pass through the network. Takes input data along with
    #array of values for group membership.


    # TODO: this needs to be updated to work with current code

    approx_posterior = self.construct_variational_posterior(gid)

    x = self.reshape(x)
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.flatten(x)
    x = self.layer1(x)
    z = tf.squeeze(approx_posterior.sample(num_samples))
    x = tf.stack([x]*num_samples)
    x = tf.concat([x, z], axis=-1)

    log_importance_weights = (
        self.z_prior.log_prob(z) - approx_posterior.log_prob(z))

    return self.out(x), log_importance_weights
"""



class NormalCNN(BaseModel):
    """
    Normal CNN for FEMNIST data.
    """

    def __init__(self, optimizer, loss_fn, num_groups, args, experiment_dir, logger):
        super(NormalCNN, self).__init__(
            optimizer, loss_fn, num_groups, args, experiment_dir, logger)

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

    def call(self, x, gid):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.layer1(x)
        return self.out(x)


class LatentFactorCNN(BaseModel):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, optimizer, loss_fn, num_groups, args, experiment_dir, logger):
        super(LatentFactorCNN, self).__init__(
            optimizer, loss_fn, num_groups, args, experiment_dir, logger)

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


    def _build_latent_space(self):
        # (num_groups, z_dim), (num_groups, z_dim), (z_dim,)
        self.z_mu, self.z_sigma, self.z_prior = latent_normal_vector(
            shape=[self.num_groups[0], self.z_dim[0]])

    def construct_variational_posterior(self, gid):
        # samples are shape (batch_size, z_dim)
        post1 = latent_vector_variational_posterior(
            self.z_mu, self.z_sigma, gid)
        return post1


    def call(self, x, gid):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.layer1(x)

        z_var_post = self.construct_variational_posterior(gid)
        kl_loss = tfd.kl_divergence(z_var_post, self.z_prior)
        self.add_loss(lambda: kl_loss)
        z = z_var_post.sample()

        x = tf.concat([x, z], axis=-1)
        return self.out(x)



class DoubleLatentCNN(BaseModel):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, optimizer, loss_fn, num_groups, args, experiment_dir, logger):
        super(DoubleLatentCNN, self).__init__(
            optimizer, loss_fn, num_groups, args, experiment_dir, logger)

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

    def _build_latent_space(self):
        # (num_groups, z_dim), (num_groups, z_dim), (z_dim,)
        self.z_mu, self.z_sigma, self.z_prior = latent_normal_vector(
            shape=[self.num_groups[0], self.z_dim[0]])
        self.z2_mu, self.z2_sigma, self.z2_prior = latent_normal_vector(
            shape=[self.num_groups[0], self.z_dim[1]])

    def construct_variational_posterior(self, gid):
        # samples are shape (batch_size, z_dim)
        post1 = latent_vector_variational_posterior(
            self.z_mu, self.z_sigma, gid)
        post2 = latent_vector_variational_posterior(
            self.z2_mu, self.z2_sigma, gid)
        return post1, post2

    def call(self, x, gid):
        z_var_post, z2_var_post = (
            self.construct_variational_posterior(gid))
        kl_loss = tfd.kl_divergence(z_var_post, self.z_prior)
        kl_loss2 = tfd.kl_divergence(z2_var_post, self.z2_prior)
        self.add_loss(lambda: kl_loss + kl_loss2)

        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)

        z2 = z2_var_post.sample()
        x = tf.concat([x, z2], axis=-1)
        x = self.layer1(x)

        z = z_var_post.sample()
        x = tf.concat([x, z], axis=-1)
        return self.out(x)





class LatentBiasCNN(BaseModel):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, optimizer, loss_fn, num_groups, args, experiment_dir, logger):
        super(LatentBiasCNN, self).__init__(
            optimizer, loss_fn, num_groups, args, experiment_dir, logger)

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
        # Turn off built in bias, turn off softmax
        self.out = Dense(62, use_bias=False)

    def _build_latent_space(self):
        # (num_groups, 62), (num_groups, 62), (62,)
        self.z_mu, self.z_sigma, self.z_prior = latent_normal_vector(
            shape=[self.num_groups[0], 62])

    def construct_variational_posterior(self, gid):
        # samples are shape (batch_size, z_dim)
        post1 = latent_vector_variational_posterior(
            self.z_mu, self.z_sigma, gid)
        return post1

    def call(self, x, gid):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.layer1(x)

        bias_posterior = self.construct_variational_posterior(gid)
        z_bias = bias_posterior.sample()
        kl_loss = tfd.kl_divergence(bias_posterior, self.z_prior)
        self.add_loss(lambda: kl_loss)

        x = self.out(x)
        return tf.nn.softmax(x + z_bias)
