import unittest
import sys
sys.path.append('src')
import hmc as hmc
import params as params
import jax.numpy as jnp
import jax.random as random
from test_helpers import random_int_uniform
from test_helpers import random_float_uniform


class TestHMC(unittest.TestCase):

    def setUp(self):
        self.key = random.PRNGKey(0)
        self.H_old = jnp.array([1.0, 2.0, 3.0])
        self.H_prime = jnp.array([0.5, 2.5, 4.0])
        self.phi_old = jnp.ones((3, 4))
        self.phi_prime = 2 * jnp.ones((3, 4))
        self.mom_old = jnp.zeros((3, 4))
        self.mom_prime = jnp.full((3, 4), 9.0)

    def test_HMC_core_accept_lower_H(self):
        # Should always accept everything when H_prime < H_old
        result = hmc.HMC_core(self.H_old,
                              self.H_prime,
                              self.phi_old,
                              self.phi_prime,
                              self.mom_old,
                              self.mom_prime,
                              self.key)
        mom_accepted = result[0]
        phi_accepted = result[1]
        mask = result[2]
        delta_H = result[3]

        # dH <0 at first index
        self.assertTrue(mask[0])
        self.assertEqual(phi_accepted.shape, self.phi_old.shape)
        self.assertEqual(mom_accepted.shape, self.mom_old.shape)

    def test_HMC_core_different_shapes(self):
        wrong_phi_old = jnp.ones((2, 4))

        self.assertRaises(ValueError, hmc.HMC_core,
                          self.H_old, self.H_prime,
                          wrong_phi_old, self.phi_prime,
                          self.mom_old, self.mom_prime,
                          self.key)

    def test_HMC_core_reject_when_higher(self):
        # If hprime is huge, we should reject everything
        H_prime_big = self.H_old + 100.0
        mom_acc, phi_acc, mask, delta_H = hmc.HMC_core(
            self.H_old, H_prime_big,
            self.phi_old, self.phi_prime,
            self.mom_old, self.mom_prime,
            self.key
        )

        self.assertTrue(jnp.all(phi_acc == self.phi_old))
        self.assertTrue(jnp.all(mom_acc == self.mom_old))
        self.assertTrue(jnp.all(mask == 0))


if __name__ == '__main__':
    unittest.main()
