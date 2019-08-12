
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions



def latent_normal_vector(shape):
    """
    Common initialization when adding a latent vector
    with a normal prior to a model. Build two trainable
    matrices of means and variances. Intended use case
    is that shape is (num_groups, z_dim).
    """

    z_mu = tf.Variable(tfd.Normal(0,0.1).sample(shape))
    z_sigma = tf.Variable(tfd.Gamma(10,10).sample(shape))
    z_prior = tfd.MultivariateNormalDiag(
        loc=np.zeros((shape[1]), dtype=np.float32),
        scale_diag=np.ones((shape[1]), dtype=np.float32))
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

def softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.math.log(tf.math.expm1(x))