import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
import numpy as np


def make_keys(N, s=0, randomize_keys=True):
    '''
    prepare keys for sampling, return array of keys
    randomizes seed by default
    if randomize = False, seed is not
         randomized for traceability/reproducibility
    returns array of N jax PRNGkeys
    has to be host side because of np.random.randint
    '''
    if isinstance(s, jnp.ndarray) and s.shape == (2,):
        master = s
    else:
        seed = int(s)
        if randomize_keys:
            seed = np.random.randint(0, N * 10**6)
        master = random.PRNGKey(seed)

    subkeys = random.split(master, N)
    return master, subkeys


def init_fields(lat_shape, seed, mom_seed, n_keys, mu, sigma, D):
    master_key = random.PRNGKey(seed)
    keys = random.split(master_key, n_keys)
    rng = partial(random.normal, shape=lat_shape, dtype=jnp.float64)
    phi_x = mu + sigma * jax.vmap(rng)(keys)

    mom_master_key = random.PRNGKey(mom_seed)
    mom_keys = random.split(mom_master_key, n_keys)
    mom_x = jax.vmap(rng)(mom_keys)

    spatial_axes = tuple(range(phi_x.ndim - D, phi_x.ndim))
    shift = phi_x.ndim - D
    return phi_x, mom_x, spatial_axes, shift


def randomize_core(keys, lat_shape, mu, sigma):
    """
    Pure JIT’d kernel
    given N keys, draws N phi-fields.
    lat_shape is static.
    """
    # vectorized normal draws
    rng = partial(random.normal, shape=lat_shape, dtype=jnp.float64)
    return jax.vmap(rng)(keys)
