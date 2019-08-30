import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Reshape, Flatten, Embedding, LSTM, Conv1D, Lambda)

from models.base_model import BaseModel
from models.model_utils import (
    init_mean_field_vectors, build_normal_variational_posterior)

tfd = tfp.distributions


# TODO: Add sampling


class NormalLSTM(BaseModel):
    """
    Normal LSTM for Shakespeare data.
    """

    def __init__(self, optimizer, loss_fn, num_groups, args, experiment_dir, logger):
        super(NormalLSTM, self).__init__(
            optimizer, loss_fn, num_groups, args, experiment_dir, logger)

    def _build_model(self):
        if self.model_size=='small':
            params=[64, 16, 16, 64]
        if self.model_size=='large':
            params=[128, 256, 256, 256]

        self.embedding = Embedding(53, params[0], input_length=80)
        self.lstm1 = LSTM(params[1], return_sequences=True)
        self.lstm2 = LSTM(params[2])
        self.dense = Dense(params[3], activation='relu')
        self.out = Dense(53, activation='softmax')

    def call(self, x, gid, gid2):
        x = self.embedding(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense(x)
        return self.out(x)


class LatentFactorLSTM(BaseModel):
    """
    Latent variable LSTM for Shakespeare data.
    """

    def __init__(self, optimizer, loss_fn, num_groups, args, experiment_dir, logger):
        super(LatentFactorLSTM, self).__init__(
            optimizer, loss_fn, num_groups, args, experiment_dir, logger)

    def _build_model(self):
        if self.model_size=='small':
            params=[64, 16, 16, 64]
        if self.model_size=='large':
            params=[128, 256, 256, 256]

        self.embedding = Embedding(53, params[0], input_length=80)
        self.lstm1 = LSTM(params[1], return_sequences=True)
        self.lstm2 = LSTM(params[2])
        self.dense = Dense(params[3], activation='relu')
        self.out = Dense(53, activation='softmax')

    def _build_latent_space(self):
        # (num_groups, z_dim), (num_groups, z_dim), (z_dim,)
        self.z_mu, self.z_sigma, self.z_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[0]])
        # (num_groups2, z_dim2), (num_groups2, z_dim2), (z_dim2,)
        self.z2_mu, self.z2_sigma, self.z2_prior = init_mean_field_vectors(
            shape=[self.num_groups[1], self.z_dim[1]])

    def construct_variational_posterior(self, gid, gid2):
        # samples are shape (batch_size, z_dim)
        post1 = build_normal_variational_posterior(
            self.z_mu, self.z_sigma, gid)
        post2 = build_normal_variational_posterior(
            self.z2_mu, self.z2_sigma, gid2)
        return post1, post2

    def call(self, x, gid, gid2):
        x = self.embedding(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense(x)

        z_var_post, z2_var_post = (
            self.construct_variational_posterior(gid, gid2))

        z = z_var_post.sample()
        z2 = z2_var_post.sample()
        kl_loss = tfd.kl_divergence(z_var_post, self.z_prior)
        kl_loss2 = tfd.kl_divergence(z2_var_post, self.z2_prior)
        self.add_loss(lambda: kl_loss + kl_loss2)

        x = tf.concat([x, z, z2], axis=1)
        return self.out(x)




class DoubleLatentLSTM(BaseModel):
    """
    Latent variable LSTM for Shakespeare data.
    """

    def __init__(self, optimizer, loss_fn, num_groups, args, experiment_dir, logger):
        super(DoubleLatentLSTM, self).__init__(
            optimizer, loss_fn, num_groups, args, experiment_dir, logger)

    def _build_model(self):
        if self.model_size=='small':
            params=[64, 16, 16, 64]
        if self.model_size=='large':
            params=[128, 256, 256, 256]

        self.embedding = Embedding(53, params[0], input_length=80)
        self.lstm1 = LSTM(params[1], return_sequences=True)
        self.lstm2 = LSTM(params[2])
        self.dense = Dense(params[3], activation='relu')
        self.out = Dense(53, activation='softmax')

    def _build_latent_space(self):
        # z1: character group, last layer
        # z2: play group, last layer
        # z3: character group, second to last layer
        # z4: play group, second to last layer
        # (num_groups, z_dim), (num_groups, z_dim), (z_dim,)
        self.z_mu, self.z_sigma, self.z_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[0]])
        self.z2_mu, self.z2_sigma, self.z2_prior = init_mean_field_vectors(
            shape=[self.num_groups[1], self.z_dim[1]])
        self.z3_mu, self.z3_sigma, self.z3_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[2]])
        self.z4_mu, self.z4_sigma, self.z4_prior = init_mean_field_vectors(
            shape=[self.num_groups[1], self.z_dim[3]])

    def construct_variational_posterior(self, gid, gid2):
        # samples are shape (batch_size, z_dim)
        post1 = build_normal_variational_posterior(
            self.z_mu, self.z_sigma, gid)
        post2 = build_normal_variational_posterior(
            self.z2_mu, self.z2_sigma, gid2)
        post3 = build_normal_variational_posterior(
            self.z3_mu, self.z3_sigma, gid)
        post4 = build_normal_variational_posterior(
            self.z4_mu, self.z4_sigma, gid2)
        return post1, post2, post3, post4

    def _double_latent_call(self, x, gid, gid2):
        """
        Forward pass through the network. Takes input data along with
        array of values for group membership.
        """
        x = self.embedding(x)
        x = self.lstm1(x)
        x = self.lstm2(x)

        z_var_post, z2_var_post, z3_var_post, z4_var_post = (
            self.construct_double_latent_posterior(gid, gid2))

        z = z_var_post.sample()
        z2 = z2_var_post.sample()
        z3 = z3_var_post.sample()
        z4 = z4_var_post.sample()
        kl_loss = tfd.kl_divergence(z_var_post, self.z_prior)
        kl_loss2 = tfd.kl_divergence(z2_var_post, self.z2_prior)
        kl_loss3 = tfd.kl_divergence(z3_var_post, self.z3_prior)
        kl_loss4 = tfd.kl_divergence(z4_var_post, self.z4_prior)
        self.add_loss(lambda: kl_loss + kl_loss2 + kl_loss3 + kl_loss4)

        x = tf.concat([x, z3, z4], axis=1)
        x = self.dense(x)
        x = tf.concat([x, z, z2], axis=1)
        return self.out(x)
