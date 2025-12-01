import unittest
import sys
sys.path.append('src')  # noqa
import hmc as hmc
import params as params
import jax.numpy as jnp
import jax.random as random
from test_helpers import random_int_uniform
from test_helpers import random_float_uniform


class TestHMCConfig(unittest.TestCase):
    def test_bad_integrator_raises(self):
        with self.assertRaises(ValueError):
            params.HMCConfig(N_steps=10,
                             eps=0.1,
                             integrator="fakes_integrator")

    def test_default_values(self):
        config = params.HMCConfig(N_steps=10, eps=0.1)
        self.assertEqual(config.integrator, "omelyan")
        self.assertEqual(config.xi, 0.1)
        self.assertEqual(config.seed, 2)
        self.assertEqual(config.N_trajectories, 1)
        self.assertTrue(config.metropolis)
        self.assertFalse(config.record_H)
        self.assertFalse(config.verbose)

    def test_invalid_N_steps_raises(self):
        with self.assertRaises(ValueError):
            params.HMCConfig(N_steps=-5, eps=0.1)

    def test_invalid_eps_raises(self):
        with self.assertRaises(ValueError):
            params.HMCConfig(N_steps=10, eps=-0.1)

    def test_invalid_seed_type_raises(self):
        with self.assertRaises(TypeError):
            params.HMCConfig(N_steps=10, eps=0.1, seed="not_an_int")

    def test_invalid_N_trajectories_raises(self):
        with self.assertRaises(ValueError):
            params.HMCConfig(N_steps=10, eps=0.1, N_trajectories=0)

    def test_invalid_metropolis_type_raises(self):
        with self.assertRaises(TypeError):
            params.HMCConfig(N_steps=10, eps=0.1, metropolis="not_a_bool")

    def test_invalid_record_H_type_raises(self):
        with self.assertRaises(TypeError):
            params.HMCConfig(N_steps=10, eps=0.1, record_H="not_a_bool")

    def test_invalid_verbose_type_raises(self):
        with self.assertRaises(TypeError):
            params.HMCConfig(N_steps=10, eps=0.1, verbose="not_a_bool")


class TestLatticeGeometry(unittest.TestCase):
    def test_dimension_inference(self):
        spacing = jnp.array([1.0, 1.0, 1.0, 1.0])
        length = jnp.array([4, 4, 4, 4])
        geom = params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        self.assertEqual(geom.D, 4)

    def test_mismatched_array_lengths_raises(self):
        spacing = jnp.array([1.0, 1.0, 1.0])
        length = jnp.array([4, 4, 4, 4])
        with self.assertRaises(ValueError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)

    def test_invalid_array_types_raises(self):
        spacing = [1.0, 1.0, 1.0]
        length = jnp.array([4, 4, 4])
        with self.assertRaises(TypeError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        spacing = jnp.array([1.0, 1.0, 1.0])
        length = [4, 4, 4]
        with self.assertRaises(TypeError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)

    def test_zero_or_negative_lengths_raises(self):
        spacing = jnp.array([1.0, 1.0, 1.0])
        length = jnp.array([4, 0, 4])
        with self.assertRaises(ValueError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        length = jnp.array([4, -2, 4])
        with self.assertRaises(ValueError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)

    def test_zero_or_negative_spacings_raises(self):
        spacing = jnp.array([1.0, 0.0, 1.0])
        length = jnp.array([4, 4, 4])
        with self.assertRaises(ValueError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        spacing = jnp.array([1.0, -1.0, 1.0])
        with self.assertRaises(ValueError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)

    def test_volume_calculation(self):
        spacing = jnp.array([1.0, 1.0, 1.0])
        length = jnp.array([4, 4, 4])
        geom = params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        self.assertEqual(geom.V, 64)

    def test_lat_shape_calculation(self):
        spacing = jnp.array([1.0, 2.0, 1.0])
        length = jnp.array([4, 4, 4])
        geom = params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        self.assertEqual(geom.lat_shape, (4, 2, 4))

    def test_invalid_dimension_raises(self):
        spacing = jnp.array([])
        length = jnp.array([])
        with self.assertRaises(ValueError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)

    def test_non_array_inputs_raises(self):
        spacing = "not_an_array"
        length = jnp.array([4, 4, 4])
        with self.assertRaises(TypeError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        spacing = jnp.array([1.0, 1.0, 1.0])
        length = "not_an_array"
        with self.assertRaises(TypeError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)

    def test_non_positive_integer_array_elements_raises(self):
        spacing = jnp.array([1, 0, 1])
        length = jnp.array([4, 4, 4])
        with self.assertRaises(ValueError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        spacing = jnp.array([1, 1, 1])
        length = jnp.array([4, -2, 4])
        with self.assertRaises(ValueError):
            params.LatticeGeometry(spacing_arr=spacing, length_arr=length)

    def test_large_dimension(self):
        D = 10
        spacing = jnp.ones(D, dtype=int)
        length = jnp.ones(D, dtype=int) * 4
        geom = params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        self.assertEqual(geom.D, D)
        self.assertEqual(geom.V, 4**D)
        self.assertEqual(geom.lat_shape, tuple([4]*D))

    def test_single_dimension(self):
        spacing = jnp.array([1])
        length = jnp.array([10])
        geom = params.LatticeGeometry(spacing_arr=spacing, length_arr=length)
        self.assertEqual(geom.D, 1)
        self.assertEqual(geom.V, 10)
        self.assertEqual(geom.lat_shape, (10,))


class TestPhi4Params(unittest.TestCase):
    def test_valid_params(self):
        params_obj = params.Phi4Params(lam=0.1, kappa=0.2)
        self.assertEqual(params_obj.lam, 0.1)
        self.assertEqual(params_obj.kappa, 0.2)


if __name__ == '__main__':
    unittest.main()
