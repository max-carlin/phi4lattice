import unittest
import sys
sys.path.append('src')
from test_helpers import random_int_uniform
from test_helpers import random_float_uniform
from layout_utils import infer_layout
import jax.numpy as jnp
from prng import make_keys
from prng import randomize_uniform_core


class TestInferLayout(unittest.TestCase):
    def test_batched(self):
        D = random_int_uniform(n=1, lower=1, upper=5)[0]

        # random spatial shape, length D
        L_array = random_int_uniform(n=D, lower=2, upper=5)
        spatial_shape = tuple(L_array)

        batch_size = random_int_uniform(n=1, lower=2, upper=10)[0]
        phi_x = jnp.ones(shape=(2, *spatial_shape))

        spatial_axes, shift = infer_layout(phi_x, D=D)

        expected_spatial_axes = tuple(range(1, D+1))
        self.assertEqual(spatial_axes, expected_spatial_axes)
        self.assertEqual(shift, 1)

    def test_singular(self):
        D = random_int_uniform(n=1, lower=1, upper=5)[0]

        # random spatial shape, length D
        L_array = random_int_uniform(n=D, lower=2, upper=5)
        spatial_shape = tuple(L_array)

        phi_x = jnp.ones(shape=spatial_shape)

        spatial_axes, shift = infer_layout(phi_x, D=D)

        expected_spatial_axes = tuple(range(0, D))
        self.assertEqual(spatial_axes, expected_spatial_axes)
        self.assertEqual(shift, 0)


if __name__ == '__main__':
    unittest.main()
