import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, Reshape, Flatten)
#from tensorflow_probability.layers import DenseVariational

from tensorflow.keras.utils import to_categorical

from models.base_model import BaseModel
from models.model_utils import (
    init_mean_field_vectors, build_normal_variational_posterior,
    latent_normal_matrix, latent_matrix_variational_posterior, softplus_inverse)

from models.multilevel_layers import MyMultilevelDense

tfd = tfp.distributions
tfpl = tfp.layers

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

    def __init__(self, *args, **kwargs):
        super(NormalCNN, self).__init__(*args, **kwargs)

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


class OneHotCNN(BaseModel):
    """
    Normal CNN for FEMNIST data.
    """

    def __init__(self, *args, **kwargs):
        super(OneHotCNN, self).__init__(*args, **kwargs)

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
        x = np.concatenate([x,to_categorical(gid, num_classes=self.num_groups[0])], axis=1)
        x = self.layer1(x)
        x = np.concatenate([x,to_categorical(gid, num_classes=self.num_groups[0])], axis=1)

        return self.out(x)



class LatentFactorCNN(BaseModel):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, *args, **kwargs):
        super(LatentFactorCNN, self).__init__(*args, **kwargs)

        # TODO: move to superclass
        #self.encoder_size = args.encoder_size

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
        self.z_mu, self.z_sigma, self.z_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[0]])

    # def _build_encoder(self):
    #     if self.encoder_size=='small':
    #         params=[4, 3, 8, 3, 32]
    #     if self.encoder_size=='large':
    #         params=[8, 3, 16, 3, 64]

    #     self.reshape = Reshape((28,28,1), input_shape=(784,))
    #     self.conv1 = Conv2D(filters=params[0], kernel_size=params[1],
    #         padding='same', activation='relu')
    #     self.pool1 = MaxPooling2D(2)
    #     self.conv2 = Conv2D(filters=params[2], kernel_size=params[3],
    #         padding='same', activation='relu')
    #     self.pool2 = MaxPooling2D(2)
    #     self.flatten = Flatten()
    #     self.layer1 = Dense(units=params[4], activation='relu')
    #     self.out = Dense(62, activation='softmax')

    def construct_variational_posterior(self, gid):
        # samples are shape (batch_size, z_dim)
        post1 = build_normal_variational_posterior(
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
        self.add_loss(tf.reduce_sum(kl_loss))
        z = z_var_post.sample()

        x = tf.concat([x, z], axis=-1)
        return self.out(x)



class LatentFactorCNN2(BaseModel):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, *args, **kwargs):
        super(LatentFactorCNN2, self).__init__(*args, **kwargs)

        # TODO: move to superclass
        #self.encoder_size = args.encoder_size

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

        self.bias_factors = Dense(62, use_bias=False)
        # TODO: should I use bias here?
        self.out = Dense(62)

    def _build_latent_space(self):
        # (num_groups, z_dim), (num_groups, z_dim), (z_dim,)
        self.z_mu, self.z_sigma, self.z_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[0]])

    def construct_variational_posterior(self, gid):
        # samples are shape (batch_size, z_dim)
        post1 = build_normal_variational_posterior(
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
        x = self.out(x)

        z_var_post = self.construct_variational_posterior(gid)
        kl_loss = tfd.kl_divergence(z_var_post, self.z_prior)
        self.add_loss(tf.reduce_sum(kl_loss))
        z = z_var_post.sample()

        bias = self.bias_factors(z)
        #x = x + self.bias
        x = x + bias
        return tf.nn.softmax(x)





class LowerLatentFactorCNN(LatentFactorCNN):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, *args, **kwargs):
        super(LowerLatentFactorCNN, self).__init__(*args, **kwargs)

    def call(self, x, gid):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)

        z_var_post = self.construct_variational_posterior(gid)
        kl_loss = tfd.kl_divergence(z_var_post, self.z_prior)
        self.add_loss(tf.reduce_sum(kl_loss))
        z = z_var_post.sample()

        x = tf.concat([x, z], axis=-1)

        x = self.layer1(x)
        return self.out(x)



class DoubleLatentCNN(BaseModel):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, *args, **kwargs):
        super(DoubleLatentCNN, self).__init__(*args, **kwargs)

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
        self.z_mu, self.z_sigma, self.z_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[0]])
        self.z2_mu, self.z2_sigma, self.z2_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[1]])

    def construct_variational_posterior(self, gid):
        # samples are shape (batch_size, z_dim)
        post1 = build_normal_variational_posterior(
            self.z_mu, self.z_sigma, gid)
        post2 = build_normal_variational_posterior(
            self.z2_mu, self.z2_sigma, gid)
        return post1, post2

    def call(self, x, gid):
        z_var_post, z2_var_post = (
            self.construct_variational_posterior(gid))
        kl_loss = tfd.kl_divergence(z_var_post, self.z_prior)
        kl_loss2 = tfd.kl_divergence(z2_var_post, self.z2_prior)
        self.add_loss(tf.reduce_sum(kl_loss) + tf.reduce_sum(kl_loss2))

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

    def __init__(self, *args, **kwargs):
        super(LatentBiasCNN, self).__init__(*args, **kwargs)

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
        self.out = Dense(62, use_bias=True)

    def _build_latent_space(self):
        # (num_groups, 62), (num_groups, 62), (62,)
        #self.z_mu, self.z_sigma, self.z_prior = init_mean_field_vectors(
        #    shape=[self.num_groups[0], 62])
        shape=[self.num_groups[0], 62]
        # Note the different initialization because directly modeling bias terms
        self.z_mu = tf.Variable(tfd.Normal(0.1,0.1).sample(shape))
        self.z_sigma = tf.Variable(tfd.Normal(0,0.01).sample(shape))

        self.z0_mu = tf.Variable(tfd.Normal(0.1,0.1).sample(shape[1]))
        self.z0_sigma = tf.Variable(tfd.Normal(0,0.01).sample(shape[1]))

        self.z_prior = tfd.MultivariateNormalDiag(
            loc=self.z0_mu, scale_diag=self.z0_sigma)

        self.z0_prior = tfd.MultivariateNormalDiag(
            loc=np.zeros((shape[1]), dtype=np.float32),
            scale_diag=np.ones((shape[1]), dtype=np.float32))

        #self.z_mu, self.z_sigma, self.z_prior = init_mean_field_vectors(
        #    shape=[self.num_groups[0], 62])

    def construct_variational_posterior(self, gid):
        # samples are shape (batch_size, z_dim)
        post1 = build_normal_variational_posterior(
            self.z_mu, self.z_sigma, gid)

        prior_post = tfd.MultivariateNormalDiag(
            loc=self.z0_mu, scale_diag=tf.nn.softplus(self.z0_sigma + softplus_inverse(1.0)))

        return post1, prior_post

    def call(self, x, gid):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.layer1(x)

        bias_posterior, prior_posterior = self.construct_variational_posterior(gid)
        z_bias = bias_posterior.sample()
        #prior_post_sample = prior_posterior.sample()
        kl_loss = tfd.kl_divergence(bias_posterior, prior_posterior)
        kl_loss_prior = tfd.kl_divergence(prior_posterior, self.z0_prior)
        self.add_loss(tf.reduce_sum(kl_loss) + tf.reduce_sum(kl_loss_prior))

        x = self.out(x)
        return tf.nn.softmax(x + z_bias)





class LatentWeightCNN(BaseModel):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, *args, **kwargs):
        super(LatentWeightCNN, self).__init__(*args, **kwargs)

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

        # # self.weight_factors = tf.Variable(tfd.Normal(0.,0.1).sample(10,256*62))
        # self.weight1_factors = Dense(units=1568*params[4], use_bias=False)

        # #self.bias = tf.Variable(tfd.Normal(0.2,0.1).sample(62))
        # self.bias1_factors = Dense(units=params[4], use_bias=False)


        # self.weight_factors = tf.Variable(tfd.Normal(0.,0.1).sample(10,256*62))
        self.weight_factors = Dense(units=params[4]*62)

        #self.bias = tf.Variable(tfd.Normal(0.2,0.1).sample(62))
        self.bias_factors = Dense(units=62)

        #self.out = tfpl.DenseFlipout(62, activation='softmax')
        #self.out = Dense(62, activation='softmax')

    # def _build_latent_space(self):
    #     # (num_groups, last_layer_units, new_layer_units)
    #     if self.model_size=='small':
    #         shape = [self.num_groups[0], 256, 62]
    #     if self.model_size=='large':
    #         shape = [self.num_groups[0], 2048, 62]

    #     self.z_mu, self.z_sigma, self.z_prior = latent_normal_matrix(
    #         shape=shape)

    # def construct_variational_posterior(self, gid):
    #     # Samples are shape (batch_size, last_layer, new_layer)
    #     post = latent_matrix_variational_posterior(
    #         self.z_mu, self.z_sigma, gid)
    #     return post

    def _build_latent_space(self):
        self.z_mu, self.z_sigma, self.z_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[0]])

        self.z2_mu, self.z2_sigma, self.z2_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[1]])

        # self.z3_mu, self.z3_sigma, self.z3_prior = init_mean_field_vectors(
        #     shape=[self.num_groups[0], 4])

        # self.z4_mu, self.z4_sigma, self.z4_prior = init_mean_field_vectors(
        #     shape=[self.num_groups[0], 16])

    def construct_variational_posterior(self, gid):
        # samples are shape (batch_size, z_dim)
        post1 = build_normal_variational_posterior(
            self.z_mu, self.z_sigma, gid)
        post2 = build_normal_variational_posterior(
            self.z2_mu, self.z2_sigma, gid)
        # post3 = build_normal_variational_posterior(
        #     self.z3_mu, self.z3_sigma, gid)
        # post4 = build_normal_variational_posterior(
        #     self.z4_mu, self.z4_sigma, gid)
        return post1, post2

    def call(self, x, gid):
        z_var_post, z2_var_post = (
          self.construct_variational_posterior(gid))
        kl_loss = tfd.kl_divergence(z_var_post, self.z_prior)
        kl_loss2 = tfd.kl_divergence(z2_var_post, self.z2_prior)
        self.add_loss(tf.reduce_sum(kl_loss) + tf.reduce_sum(kl_loss2))

        # z = tf.gather(self.z_mu, gid)
        # self.add_loss(-1*tf.reduce_sum(self.z_prior.log_prob(z)))

        # z2 = tf.gather(self.z2_mu, gid)
        # self.add_loss(-1*tf.reduce_sum(self.z2_prior.log_prob(z2)))

        # z3 = tf.gather(self.z3_mu, gid)
        # self.add_loss(-1*tf.reduce_sum(self.z3_prior.log_prob(z3)))

        # z4 = tf.gather(self.z4_mu, gid)
        # self.add_loss(-1*tf.reduce_sum(self.z4_prior.log_prob(z4)))



        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)

        x = self.layer1(x)
        # weights = self.weight1_factors(z3)
        # weights = Reshape((1568, 256))(weights)
        # x = tf.expand_dims(x, axis=-1)
        # x = tf.linalg.matmul(weights, x, transpose_a=True)
        # x = tf.squeeze(x)

        # bias = self.bias1_factors(z4)
        # #x = x + self.bias
        # x = x + bias
        # x = tf.nn.relu(x)



        z = z_var_post.sample()
        z2 = z2_var_post.sample()

        #print(tf.reduce_mean(tf.math.abs(z)))

        # # TODO: get rid of expand dims and squeeze
        # x = tf.expand_dims(x, axis=-1)
        # x = tf.linalg.matmul(z, x, transpose_a=True)
        # x = tf.squeeze(x)

        weights = self.weight_factors(z)
        weights = Reshape((256, 62))(weights)
        x = tf.expand_dims(x, axis=-1)
        x = tf.linalg.matmul(weights, x, transpose_a=True)
        x = tf.squeeze(x)

        bias = self.bias_factors(z2)
        #x = x + self.bias
        x = x + bias
        return tf.nn.softmax(x)
        #return self.out(x)







class MyLatentWeightCNN(BaseModel):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, *args, **kwargs):
        super(MyLatentWeightCNN, self).__init__(*args, **kwargs)


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
        #self.layer1 = Dense(units=params[4], activation='relu')
        self.layer1 = Dense(units=128, activation='relu')
        self.ml_dense = MyMultilevelDense(units=62, num_groups=self.num_groups[0], activation='softmax')
        #self.ml_dense = Dense(units=62, activation='softmax')

    def call(self, x, gid):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.ml_dense([x,gid])
        return x





class LatentWeightOnlyCNN(BaseModel):
    """
    Latent variable CNN for FEMNIST data.
    """

    def __init__(self, *args, **kwargs):
        super(LatentWeightOnlyCNN, self).__init__(*args, **kwargs)

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

        # # self.weight_factors = tf.Variable(tfd.Normal(0.,0.1).sample(10,256*62))
        # self.weight1_factors = Dense(units=1568*params[4], use_bias=False)

        # #self.bias = tf.Variable(tfd.Normal(0.2,0.1).sample(62))
        # self.bias1_factors = Dense(units=params[4], use_bias=False)


        # self.weight_factors = tf.Variable(tfd.Normal(0.,0.1).sample(10,256*62))
        self.weight_factors = Dense(units=params[4]*62)

        #self.bias = tf.Variable(tfd.Normal(0.2,0.1).sample(62))
        #self.bias_factors = Dense(units=62)

        #self.out = tfpl.DenseFlipout(62, activation='softmax')
        self.out = Dense(62, activation='softmax')

    # def _build_latent_space(self):
    #     # (num_groups, last_layer_units, new_layer_units)
    #     if self.model_size=='small':
    #         shape = [self.num_groups[0], 256, 62]
    #     if self.model_size=='large':
    #         shape = [self.num_groups[0], 2048, 62]

    #     self.z_mu, self.z_sigma, self.z_prior = latent_normal_matrix(
    #         shape=shape)

    # def construct_variational_posterior(self, gid):
    #     # Samples are shape (batch_size, last_layer, new_layer)
    #     post = latent_matrix_variational_posterior(
    #         self.z_mu, self.z_sigma, gid)
    #     return post

    def _build_latent_space(self):
        self.z_mu, self.z_sigma, self.z_prior = init_mean_field_vectors(
            shape=[self.num_groups[0], self.z_dim[0]])

        # self.z3_mu, self.z3_sigma, self.z3_prior = init_mean_field_vectors(
        #     shape=[self.num_groups[0], 4])

        # self.z4_mu, self.z4_sigma, self.z4_prior = init_mean_field_vectors(
        #     shape=[self.num_groups[0], 16])

    def construct_variational_posterior(self, gid):
        # samples are shape (batch_size, z_dim)
        post1 = build_normal_variational_posterior(
            self.z_mu, self.z_sigma, gid)
        # post3 = build_normal_variational_posterior(
        #     self.z3_mu, self.z3_sigma, gid)
        # post4 = build_normal_variational_posterior(
        #     self.z4_mu, self.z4_sigma, gid)
        return post1

    def call(self, x, gid):
        z_var_post = (
          self.construct_variational_posterior(gid))
        kl_loss = tfd.kl_divergence(z_var_post, self.z_prior)
        self.add_loss(tf.reduce_sum(kl_loss))

        # z = tf.gather(self.z_mu, gid)
        # self.add_loss(-1*tf.reduce_sum(self.z_prior.log_prob(z)))

        # z2 = tf.gather(self.z2_mu, gid)
        # self.add_loss(-1*tf.reduce_sum(self.z2_prior.log_prob(z2)))

        # z3 = tf.gather(self.z3_mu, gid)
        # self.add_loss(-1*tf.reduce_sum(self.z3_prior.log_prob(z3)))

        # z4 = tf.gather(self.z4_mu, gid)
        # self.add_loss(-1*tf.reduce_sum(self.z4_prior.log_prob(z4)))



        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)

        x = self.layer1(x)
        # weights = self.weight1_factors(z3)
        # weights = Reshape((1568, 256))(weights)
        # x = tf.expand_dims(x, axis=-1)
        # x = tf.linalg.matmul(weights, x, transpose_a=True)
        # x = tf.squeeze(x)

        # bias = self.bias1_factors(z4)
        # #x = x + self.bias
        # x = x + bias
        # x = tf.nn.relu(x)



        z = z_var_post.sample()

        #print(tf.reduce_mean(tf.math.abs(z)))

        # # TODO: get rid of expand dims and squeeze
        # x = tf.expand_dims(x, axis=-1)
        # x = tf.linalg.matmul(z, x, transpose_a=True)
        # x = tf.squeeze(x)

        weights = self.weight_factors(z)
        weights = Reshape((256, 62))(weights)
        x = tf.expand_dims(x, axis=-1)
        x = tf.linalg.matmul(weights, x, transpose_a=True)
        x = tf.squeeze(x)

        #bias = self.bias_factors(z2)
        #x = x + self.bias
        #x = x + bias
        #return tf.nn.softmax(x)
        return self.out(x)
