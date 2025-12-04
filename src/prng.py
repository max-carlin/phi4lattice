import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
import numpy as np
import numbers
jax.config.update("jax_enable_x64", True)


def make_keys(N_keys: int,  # number of keys to make (batch size)
              # seed or master prng key
              seed_or_key: int | jnp.ndarray | None = None,
              randomize_keys: bool = True
              ) -> tuple[jnp.ndarray, jnp.ndarray]:
    '''
    Prepare keys for sampling, returns tuple of arrays of keys;
    (master key, array of subkeys)
    where the master key is used for further splitting if needed
    and the N subkeys are used for sampling.

    Parameters
    ----------
    seed_or_key : int, PRNGKey
        * randomize_keys must be False if providing seed or key
        * keys are deterministic if seed or key is provided
    seed_or_key : None:
        * randomize_keys must be True
        * keys are random if seed_or_key is None

    Notes
    -----
    Randomization has to be host side because of np.random.randint
    (jit doesn't like it)
    '''
    if not isinstance(N_keys, numbers.Integral) or N_keys <= 0:
        raise ValueError("N_keys must be positive integer.")
    if not isinstance(randomize_keys, bool):
        raise TypeError("randomize_keys must be a boolean.")

    # deterministic path
    #   existing key path
    if isinstance(seed_or_key, jnp.ndarray) and seed_or_key.shape == (2,):
        if randomize_keys:
            raise ValueError("randomize_keys must be False when providing"
                             " a PRNGKey or seed.")
        master = seed_or_key
    #   seed path
    elif isinstance(seed_or_key, numbers.Integral):
        if randomize_keys:
            raise ValueError("randomize_keys must be False when providing"
                             " a PRNGKey or seed.")
        seed = int(seed_or_key)
        master = random.PRNGKey(seed)

    # random path
    elif seed_or_key is None:
        if randomize_keys:
            seed_or_key = np.random.randint(0, N_keys * 10**6)
        elif not randomize_keys:
            raise ValueError("Cannot have randomize_keys=False when"
                             " seed_or_key is None.")
        master = random.PRNGKey(seed_or_key)

    else:
        raise TypeError("seed_or_key must be int, PRNGKey, or None.")

    # subkey shape (N, 2)
    subkeys = random.split(master, N_keys)
    return master, subkeys


def randomize_normal_core(keys: jnp.ndarray,
                          lat_shape: tuple[int, ...],
                          mu: int | float | None = None,
                          sigma: int | float | None = None
                          ) -> jnp.ndarray:
    """
    Given N keys, draws a batch of N, independent,
    random fields from a normal distribution
    with mean mu and standard deviation sigma.
    Returns array of shape (N, *lat_shape)
    if mu and sigma are None, draws from standard normal.

    Parameters
    ----------
    keys : jnp.ndarray, shape (N, 2)
        Array of JAX PRNG keys, one per field to generate.
    lat_shape : tuple of int
        Shape of each lattice field to generate.
    mu : float or int or None, optional
        Mean of the normal distribution.
    sigma : float or int or None, optional
        Standard deviation of the normal distribution.

    Returns
    -------
    jnp.ndarray, shape (N, *lat_shape)
        Batch of N independent fields.
    """
    # vectorized normal draws
    #     partial partially evaluates random.normal
    #     so that shape and dtype are fixed
    rng = partial(random.normal, shape=lat_shape, dtype=jnp.float64)
    # standard normal if mu and sigma are None
    if mu is None:
        mu = 0.0
    if sigma is None:
        sigma = 1.0
    if not isinstance(mu, numbers.Real):
        raise TypeError("mu must be int or float or None.")
    if not isinstance(sigma, numbers.Real):
        raise TypeError("sigma must be int or float or None.")
    return mu + sigma * jax.vmap(rng)(keys)


def randomize_uniform_core(keys: jnp.ndarray,
                           lat_shape: tuple[int, ...]
                           ) -> jnp.ndarray:
    '''
    Given N keys, draws a batch of N, independent,
    random fields from a uniform distribution
    over the interval [-1, 1].
    Returns array of shape (N, *lat_shape)

    Parameters
    ----------
    keys : jnp.ndarray, shape (N, 2)
        Array of JAX PRNG keys.
    lat_shape : tuple of int
        Shape of each lattice field to generate.

    Returns
    -------
    jnp.ndarray, shape (N, *lat_shape)
        Fields drawn uniformly.
    '''
    rng = partial(random.uniform,
                  shape=lat_shape,
                  dtype=jnp.float64,
                  minval=-1.0,
                  maxval=1.0)
    return jax.vmap(rng)(keys)


def _make_traj_keys(N_trajectories: int, *,
                    seed_or_key: int | jnp.ndarray,
                    randomize_keys: bool = False
                    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    '''
    Given number of trajectories and a seed or prng key,
    returns a tuple of arrays of keys:
        (traj_master, traj_keys)
    Where:
        traj_master: prng key used as master key
        traj_keys: array of trajectory key pairs
                   of shape (N_trajectories, 2, 2)
    '''
    if not isinstance(N_trajectories, numbers.Integral) or N_trajectories <= 0:
        raise ValueError("N_trajectories must be positive integer.")
    traj_master, traj_keys = make_keys(2 * N_trajectories,
                                       seed_or_key=seed_or_key,
                                       randomize_keys=randomize_keys)
    traj_keys = traj_keys.reshape(N_trajectories, 2, 2)
    return traj_master, traj_keys
