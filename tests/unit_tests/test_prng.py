import random as random_basic
import unittest
import sys
import numpy as np
import os
import jax
import jax.numpy as jnp
import src.prng as prng


class TestMake_Keys(unittest.TestCase):
    def test_existing_key_or_seed(self):
        '''Test that existing key is used correctly
           and repeat calls with same key give same master
           and subkeys'''
        # if s is already a key, use it
        key = jax.random.PRNGKey(np.random.randint(0, 10**6))
        # master key should be the same as input key
        master, subkeys = prng.make_keys(5, s=key, randomize_keys=False)
        self.assertTrue(jnp.array_equal(master, key))
        self.assertEqual(subkeys.shape, (5, 2))
        master2, subkeys2 = prng.make_keys(5, s=key, randomize_keys=False)
        self.assertTrue(jnp.array_equal(master2, master))
        self.assertTrue(jnp.array_equal(subkeys2, subkeys))

    def test_seed_int(self):
        '''Test that integer seed produces consistent master
           and subkeys on repeat calls when randomize_keys=False'''
        seed = np.random.randint(0, 10**6)
        master, subkeys = prng.make_keys(5, s=seed, randomize_keys=False)
        master2, subkeys2 = prng.make_keys(5, s=seed, randomize_keys=False)
        self.assertTrue(jnp.array_equal(master, master2))
        self.assertTrue(jnp.array_equal(subkeys, subkeys2))


class TestRandomizeCore(unittest.TestCase):
    def test_standard_normal(self):
        '''Test that standard normal draws are correct'''
        master_key = jax.random.PRNGKey(np.random.randint(0, 10**6))
        N = np.random.randint(10**4, 10**5)
        keys = jax.random.split(master_key, N)
        lat_shape = (10, 10)
        result = prng.randomize_normal_core(keys, lat_shape)
        self.assertEqual(result.shape, (N, *lat_shape))
        # check that the mean is close to 0 and std is close to 1
        self.assertAlmostEqual(jnp.mean(result), 0.0, places=2)
        self.assertAlmostEqual(jnp.std(result), 1.0, places=2)

    def test_normal_with_mean_and_std(self):
        '''Test that normal draws with mean and std are correct'''
        master_key = jax.random.PRNGKey(np.random.randint(0, 10**6))
        N = np.random.randint(10**4, 10**5)
        keys = jax.random.split(master_key, N)
        lat_shape = (10, 10)
        mu = 2.0
        sigma = 3.0
        result = prng.randomize_normal_core(keys, lat_shape,
                                            mu=mu, sigma=sigma)
        self.assertEqual(result.shape, (N, *lat_shape))
        # check that the mean is close to mu and std is close to sigma
        self.assertAlmostEqual(jnp.mean(result), mu, places=2)
        self.assertAlmostEqual(jnp.std(result), sigma, places=2)


class TestRandomizeUniform(unittest.TestCase):
    def test_uniform_draws(self):
        '''Test that uniform draws are correct'''
        master_key = jax.random.PRNGKey(np.random.randint(0, 10**6))
        N = np.random.randint(10**4, 10**5)
        keys = jax.random.split(master_key, N)
        lat_shape = (10, 10)
        result = prng.randomize_uniform_core(keys, lat_shape)
        self.assertEqual(result.shape, (N, *lat_shape))
        # check that the min is close to -1 and max is close to 1
        self.assertAlmostEqual(jnp.min(result), -1.0, places=2)
        self.assertAlmostEqual(jnp.max(result), 1.0, places=2)
        self.assertAlmostEqual(jnp.mean(result), 0.0, places=2)


if __name__ == '__main__':
    unittest.main()
    sys.exit(0)
