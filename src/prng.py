import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
import numpy as np


def make_keys(N: int,   # number of keys to make (batch size)
              s: int = 0,  # seed or master prng key
              randomize_keys: bool = True
              ) -> tuple[jnp.ndarray, jnp.ndarray]:
    '''
    prepare keys for sampling, returns tuple of arrays of keys;
    (master key, array of subkeys)
    where the master key is used for further splitting if needed
    and the N subkeys are used for sampling

    if s is int, uses as seed
    if s is PRNGKey, uses as master key
    if randomize = True
        randomizes seed (default behavior)
    if randomize = False, seed is not
        randomized for traceability/reproducibility

    note: randomization has to be host side because of np.random.randint
    '''
    # if s is already a key, use it
    if isinstance(s, jnp.ndarray) and s.shape == (2,):
        master = s
    # if s is
    else:
        seed = int(s)
        if randomize_keys:
            seed = np.random.randint(0, N * 10**6)
        master = random.PRNGKey(seed)
    # subkey shape (N, 2)
    subkeys = random.split(master, N)
    return master, subkeys


def _randomize_core(keys: jnp.ndarray,
                   lat_shape: tuple[int, ...],
                   mu: float = None,
                   sigma: float = None
                   ) -> jnp.ndarray:
    """
    Given N keys, draws a batch of N, independent,
    random fields from a normal distribution
    with mean mu and standard deviation sigma.
    Returns array of shape (N, *lat_shape)
    if mu and sigma are None, draws from standard normal
    """
    # vectorized normal draws
    #     partial partially evaluates random.normal
    #     so that shape and dtype are fixed
    rng = partial(random.normal, shape=lat_shape, dtype=jnp.float64)
    # standard normal if mu and sigma are None
    if mu is None and sigma is None:
        return jax.vmap(rng)(keys)
    # else draw from normal with mean mu and std sigma
    elif mu is not None and sigma is not None:
        return mu + sigma * jax.vmap(rng)(keys)


def randomize_uniform_core(keys: jnp.ndarray,
                            lat_shape: tuple[int, ...]
                            ) -> jnp.ndarray:
    '''
    Given N keys, draws a batch of N, independent,
    random fields from a uniform distribution
    over the interval [-1, 1].
    Returns array of shape (N, *lat_shape)'''
    rng = partial(random.uniform,
                    shape=lat_shape,
                    dtype=jnp.float64,
                    minval=-1.0,
                    maxval=1.0)
    return jax.vmap(rng)(keys)


# I (sevio) don't think we need this function
# and if we do, it should probably go in lattice.py
# so I didn't add tests for it
def init_fields(lat_shape,
                seed,
                mom_seed,
                n_keys,
                mu,
                sigma,
                D):
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
