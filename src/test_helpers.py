import numpy as np
import jax.numpy as jnp


def random_int_uniform(n=25, lower=-1000, upper=1000, seed=None):
    """Generate a list of n random integers
    between lower and upper (inclusive).
    Args:
        n (int): Number of random integers to generate.
        lower (int): Lower bound for random integers.
        upper (int): Upper bound for random integers.
        seed (int): Seed for the random number generator.
    Returns:
        list: List of n random integers."""
    """
    Return a Python list of n random integers in [lower, upper], fast.
    """
    if seed is None:
        seed = np.random.randint(0, 10**6)
    rng = np.random.default_rng(seed)
    int_list = rng.integers(lower, upper,
                            size=n, endpoint=True,
                            dtype=np.int64).tolist()
    return int_list


def random_float_uniform(n=25, lower=-1000, upper=1000, seed=None):
    """Generate a list of n random floats
    between lower and upper (inclusive).
    Args:
        n (int): Number of random floats to generate.
        lower (float): Lower bound for random floats.
        upper (float): Upper bound for random floats.
        seed (int): Seed for the random number generator.
    Returns:
        list: List of n random floats."""
    """
    Return a Python list of n random floats in [lower, upper], fast.
    """
    if seed is None:
        seed = np.random.randint(0, 10**6)
    rng = np.random.default_rng(seed)
    float_list = rng.uniform(lower, upper, size=n).tolist()
    return float_list


def create_ising_field(L_array: jnp.ndarray,
                       seed=0,
                       batch_size: int = 1):
    """Create a random Ising field configuration."""
    D = len(L_array)
    lat_shape = tuple(L_array.tolist())
    rng = np.random.default_rng(seed)
    if batch_size == 1:
        sigma_x = jnp.array(rng.choice([-1, 1], size=lat_shape))
    else:
        sigma_x = jnp.array(rng.choice([-1, 1], size=(batch_size, *lat_shape)))

    return sigma_x


def create_zero_action_ising_field(L_array: jnp.ndarray,
                                   seed=0,
                                   batch_size: int = 1):
    lat_shape = tuple(L_array.tolist())
    V = int(jnp.prod(L_array))

    if V % 2 != 0:
        raise ValueError("Lattice volume must be even for zero-action config.")
    # create balanced config with half +1 and half -1
    rng = np.random.default_rng(seed)
    max_tries = 10**6
