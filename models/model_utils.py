
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

# TODO: can probably collapse into 2 functions

def latent_normal_vector(shape):
    """
    Common initialization when adding a latent vector
    with a normal prior to a model. Build two trainable
    matrices of means and variances. Intended use case
    is that shape is (num_groups, z_dim).
    """
    z_mu = tf.Variable(tfd.Normal(0,0.1).sample(shape))
    z_sigma = tf.Variable(tfd.Normal(0,0.1).sample(shape))
    z_prior = tfd.MultivariateNormalDiag(
        loc=np.zeros((shape[1]), dtype=np.float32),
        scale_diag=np.ones((shape[1]), dtype=np.float32))
    return z_mu, z_sigma, z_prior

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

def latent_vector_variational_posterior(z_mu, z_sigma, gid):
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