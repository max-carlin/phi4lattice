import unittest
import sys
import numpy as np
sys.path.append('src')  # noqa
import metropolis_mc
import params as params
import jax.numpy as jnp
import jax.random as random
from test_helpers import random_int_uniform
from test_helpers import random_float_uniform
from test_helpers import create_zero_action_ising_field
from test_helpers import create_ising_field


class TestMetropolis_MC(unittest.TestCase):

    def test_metropolis_accept_lower_S(self):
        seed = np.random.randint(0, 10**6)
        key = random.PRNGKey(seed)
        r_key, batch_key = random.split(key)
        # Should always accept when S_new < S_old
        D = random.randint(batch_key, shape=(), minval=1, maxval=5)
        l_array = random_int_uniform(n=D, lower=2, upper=5, seed=seed)
        l_array = jnp.array(l_array)
        batch_size = random.randint(batch_key, shape=(), minval=1, maxval=5)
        S_old = jnp.array(random_float_uniform(n=batch_size,
                                               lower=5,
                                               upper=10.0))
        S_new = jnp.array(random_float_uniform(n=batch_size,
                                               lower=0.1,
                                               upper=4.9))
        sigma_old = create_ising_field(L_array=l_array,
                                       batch_size=batch_size,
                                       seed=seed)
        sigma_new = create_ising_field(L_array=l_array,
                                       batch_size=batch_size,
                                       seed=seed+1)
        sigma_accepted, mask, delta_S = metropolis_mc.Metropolis(S_old,
                                                                 S_new,
                                                                 sigma_old,
                                                                 sigma_new,
                                                                 r_key)
        self.assertTrue(jnp.all(mask))
        self.assertTrue(jnp.array_equal(sigma_accepted, sigma_new))
        self.assertTrue(jnp.all(delta_S < 0))

    def test_metropolis_equal_S(self):
        seed = np.random.randint(0, 10**6)
        key = random.PRNGKey(seed)
        r_key, batch_key = random.split(key)
        # Should accept with probability 1 when S_new == S_old
        D = random.randint(batch_key, shape=(), minval=1, maxval=5)
        l_array = random_int_uniform(n=D, lower=2, upper=5, seed=seed)
        l_array = jnp.array(l_array)
        batch_size = random.randint(batch_key, shape=(), minval=1, maxval=5)
        S_old = jnp.array(random_float_uniform(n=batch_size,
                                               lower=0.1,
                                               upper=10.0))
        S_new = S_old
        sigma_old = create_ising_field(L_array=l_array,
                                       batch_size=batch_size,
                                       seed=seed)
        sigma_new = create_ising_field(L_array=l_array,
                                       batch_size=batch_size,
                                       seed=seed+1)
        sigma_accepted, mask, delta_S = metropolis_mc.Metropolis(S_old,
                                                                 S_new,
                                                                 sigma_old,
                                                                 sigma_new,
                                                                 r_key)
        self.assertTrue(jnp.all(mask))
        self.assertTrue(jnp.array_equal(sigma_accepted, sigma_new))
        self.assertTrue(jnp.all(delta_S == 0))


if __name__ == '__main__':
    unittest.main()
