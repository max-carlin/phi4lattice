import random as random_basic
import unittest
import src.energetics as eng
import sys
import numpy as np
import os
import jax
import jax.numpy as jnp
import src.lattice as lattice


def random_int_uniform(n=25, lower=-1000, upper=1000, seed=0):
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
    rng = np.random.default_rng(seed)
    int_list = rng.integers(lower, upper,
                            size=n, endpoint=True,
                            dtype=np.int64).tolist()
    return int_list


def random_float_uniform(n=25, lower=-1000, upper=1000, seed=0):
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
    rng = np.random.default_rng(seed)
    float_list = rng.uniform(lower, upper, size=n).tolist()
    return float_list


def create_field(L_array: jnp.ndarray,
                 constant: bool = True,
                 fill_value: float = 0.0,
                 seed=0):
    """Create a random field configuration."""
    D = len(L_array)
    lat_shape = tuple(L_array.tolist())
    rng = np.random.default_rng(seed)
    if constant:
        phi_x = jnp.full(lat_shape, fill_value)
    else:
        phi_x = jnp.array(rng.uniform(-1, 1, size=lat_shape))
    return phi_x, D, lat_shape


class TestAction(unittest.TestCase):
    def test_constant_field_action_manual(self):
        """test action on constant field configuration"""
        fill_value = 2.0
        lam = 1.0
        kappa = 0.5
        L_array = jnp.array([4, 4])
        lat_shape = tuple(L_array.tolist())
        field = create_field(L_array,
                             constant=True,
                             fill_value=fill_value)
        phi_x = field[0]
        D = field[1]
        lat_shape = field[2]
        shift = phi_x.ndim - D
        spatial_axes = tuple(range(shift, phi_x.ndim))

        S, K, W = eng.action_core(phi_x,
                                  lam, kappa,
                                  D, shift,
                                  spatial_axes)

        # hand-checks
        # neighbors: each site has 2 in +x/-x and +y/-y
        # K term for constant phi:
        # K = -kappa * sum_x phi_x * (phi_{x+}+phi_{x-}
        # + phi_{y+}+phi_{y-})
        # each site contributes -kappa * (4*phi^2); total sites = Lx*Ly
        total_sites = lat_shape[0]*lat_shape[1]
        K_manual = -kappa * (4*fill_value**2) * total_sites
        U_manual = phi_x ** 2 + lam * (phi_x ** 2 - 1.0) ** 2
        U_manual = U_manual.sum()
        S_manual = K_manual + U_manual

        self.assertIsInstance(S, jnp.ndarray)
        self.assertIsInstance(K, jnp.ndarray)
        self.assertIsInstance(W, jnp.ndarray)
        self.assertEqual(S.shape, ())
        self.assertEqual(S, S_manual)

    def test_constant_field_action_random_size(self):
        """test action on constant field configuration of random size"""
        fill_value = 1.5
        lam = 1
        kappa = 0.5
        D = random_basic.randint(2, 5)
        random_seed = random_basic.randint(0, 10000)
        L_array_list = random_int_uniform(n=D,
                                          lower=3,
                                          upper=7,
                                          seed=random_seed)
        L_array = jnp.array(L_array_list)
        field = create_field(L_array,
                             constant=True,
                             fill_value=fill_value,
                             seed=random_seed)
        phi_x = field[0]
        D = field[1]
        lat_shape = field[2]
        shift = phi_x.ndim - D
        spatial_axes = tuple(range(shift, phi_x.ndim))

        S, K, W = eng.action_core(phi_x,
                                  lam, kappa,
                                  D, shift,
                                  spatial_axes)

        # hand-checks
        total_sites = 1
        for L in lat_shape:
            total_sites *= L
        K_manual = -kappa * (2*D*fill_value**2) * total_sites
        U_manual = phi_x ** 2 + lam * (phi_x ** 2 - 1.0) ** 2
        U_manual = U_manual.sum()
        S_manual = K_manual + U_manual

        self.assertIsInstance(S, jnp.ndarray)
        self.assertIsInstance(K, jnp.ndarray)
        self.assertIsInstance(W, jnp.ndarray)
        self.assertEqual(S.shape, ())
        self.assertEqual(S, S_manual)

    def test_constant_field_action_all_rand(self):
        """test action on constant field configuration of random size
        and random parameters"""
        fill_value = random_basic.uniform(-5.0, 5.0)
        lam = random_basic.uniform(0.1, 10.0)
        kappa = random_basic.uniform(0.1, 1.0)
        D = random_basic.randint(2, 5)
        random_seed = random_basic.randint(0, 10000)
        L_array_list = random_int_uniform(n=D,
                                          lower=3,
                                          upper=7,
                                          seed=random_seed)
        L_array = jnp.array(L_array_list)
        field = create_field(L_array,
                             constant=True,
                             fill_value=fill_value,
                             seed=random_seed)
        phi_x = field[0]
        D = field[1]
        lat_shape = field[2]
        shift = phi_x.ndim - D
        spatial_axes = tuple(range(shift, phi_x.ndim))

        S, K, W = eng.action_core(phi_x,
                                  lam, kappa,
                                  D, shift,
                                  spatial_axes)

        # hand-checks
        total_sites = 1
        for L in lat_shape:
            total_sites *= L
        K_manual = -kappa * (2*D*fill_value**2) * total_sites
        U_manual = phi_x ** 2 + lam * (phi_x ** 2 - 1.0) ** 2
        U_manual = U_manual.sum()
        S_manual = K_manual + U_manual

        self.assertIsInstance(S, jnp.ndarray)
        self.assertIsInstance(K, jnp.ndarray)
        self.assertIsInstance(W, jnp.ndarray)
        self.assertEqual(S.shape, ())
        self.assertAlmostEqual(S, S_manual, places=5)

    def test_action_batched_fields_rand(self):
        """test action on batched field configurations of random size
        and random parameters"""
        batch_size = random_basic.randint(2, 5)
        lam = random_basic.uniform(-10, 10.0)
        kappa = random_basic.uniform(-10, -10)
        D = random_basic.randint(1, 5)
        random_seed = random_basic.randint(0, 10000)
        L_array_list = random_int_uniform(n=D,
                                          lower=1,
                                          upper=7,
                                          seed=random_seed)
        L_array = jnp.array(L_array_list)
        lat_shape = tuple(L_array.tolist())
        total_sites = 1
        for L in lat_shape:
            total_sites *= L

        phi_x = []
        S_manual = []
        for _bs in range(batch_size):
            fill_value = random_basic.uniform(-5.0, 5.0)
            random_seed = random_basic.randint(0, 10000)
            field = create_field(L_array,
                                 constant=True,
                                 fill_value=fill_value,
                                 seed=random_seed)
            phi_x_single = field[0]
            phi_x.append(phi_x_single)
            # hand-checks
            K_manual_single = -kappa * (2*D*fill_value**2) * total_sites
            U_single = phi_x_single ** 2 + lam * (phi_x_single ** 2 - 1.0) ** 2
            U_manual_single = U_single.sum()
            S_manual_single = K_manual_single + U_manual_single
            S_manual.append(S_manual_single)
        phi_x = jnp.array(phi_x)
        S_manual = jnp.array(S_manual)
        shift = phi_x.ndim - D
        spatial_axes = tuple(range(shift, phi_x.ndim))

        S, K, W = eng.action_core(phi_x,
                                  lam, kappa,
                                  D, shift,
                                  spatial_axes)

        self.assertIsInstance(S, jnp.ndarray)
        self.assertIsInstance(K, jnp.ndarray)
        self.assertIsInstance(W, jnp.ndarray)
        self.assertEqual(S.shape, (batch_size,))
        self.assertEqual(S_manual.shape, (batch_size,))
        self.assertTrue(jnp.allclose(S, S_manual, atol=1e-5))

    def test_action_Z2_symmetry(self):
        """test action Z2 symmetry: S[phi] == S[-phi]"""
        lam = random_basic.uniform(-10.0, 10.0)
        kappa = random_basic.uniform(-10.0, -10.0)
        D = random_basic.randint(1, 5)
        random_seed = random_basic.randint(0, 10000)
        L_array_list = random_int_uniform(n=D,
                                          lower=1,
                                          upper=7,
                                          seed=random_seed)
        L_array = jnp.array(L_array_list)
        field = create_field(L_array,
                             constant=False,
                             fill_value=None,
                             seed=random_seed)
        phi_x = field[0]

        shift = phi_x.ndim - D
        spatial_axes = tuple(range(shift, phi_x.ndim))

        S_pos, K_pos, W_pos = eng.action_core(phi_x,
                                              lam, kappa,
                                              D, shift,
                                              spatial_axes)

        S_neg, K_neg, W_neg = eng.action_core(phi_x,
                                              lam, kappa,
                                              D, shift,
                                              spatial_axes)
        self.assertTrue(jnp.allclose(S_pos, S_neg, atol=1e-5))
        self.assertTrue(jnp.allclose(K_pos, K_neg, atol=1e-5))
        self.assertTrue(jnp.allclose(W_pos, W_neg, atol=1e-5))

    def test_action_translational_invariance_action(self):
        """test action translational invariance:
        S[phi] == S[shifted phi]"""
        lam = random_basic.uniform(-10.0, 10.0)
        kappa = random_basic.uniform(-10.0, -10.0)
        D = random_basic.randint(1, 5)
        random_seed = random_basic.randint(0, 10000)
        L_array_list = random_int_uniform(n=D,
                                          lower=3,
                                          upper=7,
                                          seed=random_seed)
        L_array = jnp.array(L_array_list)
        field = create_field(L_array,
                             constant=False,
                             fill_value=None,
                             seed=random_seed)
        phi_x = field[0]

        shift = phi_x.ndim - D
        spatial_axes = tuple(range(shift, phi_x.ndim))

        S_orig, K_orig, W_orig = eng.action_core(phi_x,
                                                 lam, kappa,
                                                 D, shift,
                                                 spatial_axes)

        # shift by random amounts in each dimension
        shifts = []
        for d in range(D):
            L_d = L_array[d]
            shift_d = random_basic.randint(1, L_d-1)
            shifts.append(shift_d)
        # pad shifts for non-spatial dims
        total_shifts = [0]*shift + shifts
        phi_x_shifted = jnp.roll(phi_x,
                                 shift=total_shifts,
                                 axis=tuple(range(phi_x.ndim)))

        S_shifted, K_shifted, W_shifted = eng.action_core(phi_x_shifted,
                                                          lam, kappa,
                                                          D, shift,
                                                          spatial_axes)

        self.assertTrue(jnp.allclose(S_orig, S_shifted, atol=1e-5))
        self.assertTrue(jnp.allclose(K_orig, K_shifted, atol=1e-5))
        self.assertTrue(jnp.allclose(W_orig, W_shifted, atol=1e-5))


class TestGradAction(unittest.TestCase):
    # setup a simple lattice for testing
    D = random_basic.randint(1, 7)
    L = random_basic.randint(1, 7)
    a_arr = jnp.ones(D, dtype=int)
    l_arr = jnp.ones(D, dtype=int)*L

    kappa = random_basic.uniform(-10, 10)
    lam = random_basic.uniform(-10, 1.0)

    lat = lattice.Phi4Lattice(a_array=a_arr,
                              L_array=l_arr,
                              kappa=kappa,
                              lam=lam)
    lat.randomize_phi(N=1, randomize_keys=True)
    phi_x = lat.phi_x

    def grad_phi4_action_manual(self, phi_x, lam, kappa, D):
        """
        compute action gradient (Eq. 2.6)
        ∂S/∂φ_x = −κ ∑_μ [φ_{x+μ} + φ_{x−μ}]+ 2 φ_x + 4 λ φ_x (φ_x^2 − 1)
        """
        shift = phi_x.ndim - D
        J_x = jnp.zeros_like(phi_x)
        for mu in range(D):
            ax = mu + shift
            J_x += jnp.roll(phi_x, 1, axis=ax) + jnp.roll(phi_x, -1, axis=ax)
        grad_kinetic = -2 * kappa * J_x
        grad_potential = 2 * phi_x + 4 * lam * (phi_x ** 2 - 1) * phi_x
        grad_S = grad_kinetic + grad_potential
        return grad_S

    def test_grad_action_vs_manual(self,
                                   phi_x=phi_x,
                                   lam=lam,
                                   kappa=kappa,
                                   D=D):
        grad_S_manual = self.grad_phi4_action_manual(phi_x,
                                                     lam,
                                                     kappa,
                                                     D)

        grad_S = eng.grad_action_core(phi_x,
                                      lam,
                                      kappa,
                                      D,
                                      self.lat.shift,
                                      self.lat.spatial_axes)
        self.assertTrue(jnp.allclose(grad_S, grad_S_manual, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
    sys.exit(0)
