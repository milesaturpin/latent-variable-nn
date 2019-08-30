
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

# TODO: can probably collapse into 2 functions

def init_mean_field_vectors(shape, fixed_prior=True,
    mu_initializer='random_normal', sigma_initializer='random_normal'):
    """
    Common initialization when adding a vector
    with a normal prior to a model. Build two trainable
    matrices of means and variances. Intended use case
    is that shape is (num_groups, z_dim).

    Offers option to have fixed prior or not. Ofter we can
    just use a fixed prior because the layers that the random
    variable is connected to "absorbs" the mean and variance
    parameters.
    """
    mu_init = tf.keras.initializers.get(mu_initializer)
    sigma_init = tf.keras.initializers.get(sigma_initializer)
    # z_mu = tf.Variable(tfd.Normal(0,0.1).sample(shape))
    # z_sigma = tf.Variable(tfd.Normal(0,0.1).sample(shape))

    z_mu = mu_init(shape)
    z_sigma = sigma_init(shape)

    if fixed_prior:
        z_prior = tfd.MultivariateNormalDiag(
            loc=np.zeros((shape[1]), dtype=np.float32),
            scale_diag=np.ones((shape[1]), dtype=np.float32))
        #print(z_prior)
        return z_mu, z_sigma, z_prior
    else:
        z0_mu, z0_sigma, z0_prior = init_mean_field_vectors(shape[-1], fixed_prior=True)
        z_prior = tfd.MultivariateNormalDiag(
            loc=z0_mu, scale_diag=z0_sigma)
        return z_mu, z_sigma, z_prior, z0_mu, z0_sigma, z0_prior


def build_normal_variational_posterior(z_mu, z_sigma, gid):
    """
    Index into matrix to get latent vector for group.
    Performed in batches; gid is shape (batch_size, 1).
    Returns a tensorflow distribution object whose samples
    are of shape (num_samples, batch_size, z_dim).
    """
    # mu and sigma have shape (batch_size, z_dim)
    mu = tf.gather(z_mu, gid)
    sigma = tf.gather(z_sigma, gid)
    return tfd.MultivariateNormalDiag(
        loc=mu, scale_diag=tf.nn.softplus(sigma + softplus_inverse(1.0)))




def latent_normal_matrix(shape):
    """
    Common initialization when adding a latent weight matrix
    with a normal prior to a model. Build two trainable
    tensors of means and variances. Intended use case
    is that shape is (num_groups, z_dim1, z_dim2).
    """
    z_mu = tf.Variable(tfd.Normal(0,0.1).sample(shape))
    z_sigma = tf.Variable(tfd.Normal(0,0.1).sample(shape))
    z_prior = tfd.MultivariateNormalDiag(
        loc=np.zeros((shape[1], shape[2]), dtype=np.float32),
        scale_diag=np.ones((shape[1], shape[2]), dtype=np.float32))
    # Necessary to reinterpret batch dims as event dims
    z_prior = tfd.Independent(z_prior, reinterpreted_batch_ndims=1)
    return z_mu, z_sigma, z_prior



def latent_matrix_variational_posterior(z_mu, z_sigma, gid):
    """
    Index into tensor to get latent matrix for group.
    Performed in batches; gid is shape (batch_size, 1).
    Returns a tensorflow distribution object whose samples
    are of shape (num_samples, batch_size, z_dim1, z_dim2).
    """
    # mu and sigma have shape (batch_size, z_dim1, z_dim2)
    mu = tf.gather(z_mu, gid)
    sigma = tf.gather(z_sigma, gid)
    post = tfd.MultivariateNormalDiag(
        loc=mu, scale_diag=tf.nn.softplus(sigma + softplus_inverse(1.0)))
    # Necessary to reinterpret batch dims as event dims
    post = tfd.Independent(post, reinterpreted_batch_ndims=1)
    return post

def softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.math.log(tf.math.expm1(x))