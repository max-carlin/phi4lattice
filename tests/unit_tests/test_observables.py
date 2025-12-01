
import random as random_basic
import unittest
import sys
sys.path.append('src')
from observables import magnetization
import sys
import numpy as np
import os
import jax
import jax.numpy as jnp
import lattice as lattice
from test_helpers import random_int_uniform
from test_helpers import random_float_uniform


class TestMagnetization(unittest.TestCase):
    # setup a lattice for testing
    D = random_basic.randint(1, 5)
    L = random_basic.randint(1, 5)
    a_arr = jnp.ones(D, dtype=int)
    l_arr = jnp.ones(D, dtype=int)*L
    n_fields = random_basic.randint(1, 5)
    lat = lattice.Phi4Lattice(spacing_arr=a_arr,
                              length_arr=l_arr,
                              kappa=0.1,
                              lam=0.1,
                              n_keys=n_fields)

    def test_zero_field(self):
        self.lat.constant_phi(0.0)
        phi_x = self.lat.phi_x
        m = magnetization(phi_x, self.D)
        m_manual = jnp.zeros_like(m)
        self.assertTrue(jnp.allclose(m, m_manual, atol=1e-5))

    def test_constant_field(self):
        c = random_basic.uniform(-10.0, 10.0)
        self.lat.constant_phi(c)
        phi_x = self.lat.phi_x
        V = self.lat.geom.V
        m = magnetization(phi_x, self.D)
        m_manual = jnp.full_like(m, c*V, dtype=jnp.float64)
        self.assertTrue(jnp.allclose(m, m_manual, atol=1e-2))

    def test_random_uniform_field(self):
        self.lat.randomize_phi(N_fields=self.n_fields,
                               dist='uniform',
                               randomize_keys=True)
        phi_x = self.lat.phi_x
        m = magnetization(phi_x, self.D)
        m_manual = jnp.zeros_like(m)
        # std for uniform()
        var_uniform = 1.0/3.0
        m_var = self.lat.geom.V * var_uniform
        m_sigma = jnp.sqrt(m_var)
        # check that magnetizations are within 5 sigma of 0
        self.assertTrue(jnp.allclose(m, m_manual, atol=5*m_sigma))


# class TestBinderCumulant(unittest.TestCase):
#     # setup a simple lattice for testing
#     D=random.randint(1,7)
#     L=random.randint(1,7)
#     a_arr = jnp.ones(D, dtype = int)
#     l_arr = jnp.ones(D, dtype = int)*L
#     n_fields = random.randint(10,20)
#     lat = lattice.Phi4Lattice(a_array=a_arr,
#                               L_array= l_arr,
#                               kappa=0.1,
#                               lam=0.1,
#                               n_keys=n_fields)

#     def test_binder_cumulant_constant_field(self):
#         c = random.uniform(-10.0,10.0)
#         self.lat._constant_phi(c)
#         phi_x = self.lat.phi_x
#         V = self.lat.V
#         m = magnetization(phi_x, self.D)
#         m2 = jnp.mean(m**2)
#         m4 = jnp.mean(m**4)
#         B_manual = 1.0 - (m4 / (3.0 * m2**2))
#         # since all fields are constant, B_manual should be zero
#         self.assertAlmostEqual(B_manual, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
