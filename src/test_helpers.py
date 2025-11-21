import numpy as np


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
