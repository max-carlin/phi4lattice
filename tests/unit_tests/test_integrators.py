import random as random_basic
import unittest
import sys
sys.path.append('src')
import numpy as np
import os
import jax
import jax.numpy as jnp
from integrators import omelyan_integrator
from integrators import leapfrog_integrator
from test_helpers import random_int_uniform
from test_helpers import random_float_uniform
jax.config.update("jax_enable_x64", True)


def make_harmonic_energy_fns(omega: float = 1.0):
    """
    Helper to build (S_Fn, grad_S_Fn, H_kinetic_Fn)
    for a 1D harmonic oscillator.
    """
    def S_Fn(phi_x):
        # potential: 1/2 * omega^2 * phi^2
        return 0.5 * omega**2 * jnp.sum(phi_x**2)

    def grad_S_Fn(phi_x):
        # grad wrt phi: omega^2 * phi
        return omega**2 * phi_x

    def H_kinetic_Fn(mom_x):
        # kinetic: 1/2 * p^2
        return 0.5 * jnp.sum(mom_x**2)

    return S_Fn, grad_S_Fn, H_kinetic_Fn


class TestOmelyanIntegrator(unittest.TestCase):
    def test_omelyan_is_time_reversible(self):
        omega = random_float_uniform(n=1, lower=-5.0, upper=5.0)[0]
        S_Fn, grad_S_Fn, H_kinetic_Fn = make_harmonic_energy_fns(omega=omega)
        phi_0 = random_float_uniform(n=1, lower=-10.0, upper=10.0)[0]
        mom_0 = random_float_uniform(n=1, lower=-10.0, upper=10.0)[0]
        eps = random_float_uniform(n=1, lower=0.01, upper=0.1)[0]
        xi = random_float_uniform(n=1, lower=0.01, upper=0.1)[0]
        N_steps = random_int_uniform(n=1, lower=10, upper=50)[0]
        # forward integration
        mom_f, phi_f = omelyan_integrator(mom_0,
                                          phi_0,
                                          S_Fn=S_Fn,
                                          grad_S_Fn=grad_S_Fn,
                                          H_kinetic_Fn=H_kinetic_Fn,
                                          eps=eps,
                                          xi=xi,
                                          N_steps=N_steps,
                                          record_H=False)
        # backward integration
        mom_b, phi_b = omelyan_integrator(-mom_f,
                                          phi_f,
                                          S_Fn=S_Fn,
                                          grad_S_Fn=grad_S_Fn,
                                          H_kinetic_Fn=H_kinetic_Fn,
                                          eps=eps,
                                          xi=xi,
                                          N_steps=N_steps,
                                          record_H=False)
        # expect to recover initial conditions
        self.assertAlmostEqual(phi_0, phi_b, places=10)
        self.assertAlmostEqual(mom_0, -mom_b, places=10)

    def test_omelyan_energy_conservation(self):
        omega = random_float_uniform(n=1, lower=-5.0, upper=5.0)[0]
        S_Fn, grad_S_Fn, H_kinetic_Fn = make_harmonic_energy_fns(omega=omega)
        phi_0 = random_float_uniform(n=1, lower=-10.0, upper=10.0)[0]
        mom_0 = random_float_uniform(n=1, lower=-10.0, upper=10.0)[0]
        eps = random_float_uniform(n=1, lower=0.01, upper=0.1)[0]
        xi = random_float_uniform(n=1, lower=0.01, upper=0.1)[0]
        N_steps = random_int_uniform(n=1, lower=10, upper=50)[0]
        # run integrator with energy recording
        mom_f, phi_f, H_hist = omelyan_integrator(mom_0,
                                                  phi_0,
                                                  S_Fn=S_Fn,
                                                  grad_S_Fn=grad_S_Fn,
                                                  H_kinetic_Fn=H_kinetic_Fn,
                                                  eps=eps,
                                                  xi=xi,
                                                  N_steps=N_steps,
                                                  record_H=True)
        # check that energy is approximately conserved
        H_initial = H_hist[0]
        max_dev = float(jnp.max(jnp.abs(H_hist - H_initial)))
        rel_dev = max_dev / float(jnp.abs(H_initial))
        self.assertLessEqual(rel_dev, 10*eps**2)


class TestLeapfrogIntegrator(unittest.TestCase):
    def test_leapfrog_is_time_reversible(self):
        omega = random_float_uniform(n=1, lower=-5.0, upper=5.0)[0]
        S_Fn, grad_S_Fn, H_kinetic_Fn = make_harmonic_energy_fns(omega=omega)
        phi_0 = random_float_uniform(n=1, lower=-10.0, upper=10.0)[0]
        mom_0 = random_float_uniform(n=1, lower=-10.0, upper=10.0)[0]
        eps = random_float_uniform(n=1, lower=0.01, upper=0.1)[0]
        N_steps = random_int_uniform(n=1, lower=10, upper=50)[0]
        # forward integration
        mom_f, phi_f = leapfrog_integrator(mom_0,
                                           phi_0,
                                           S_Fn=S_Fn,
                                           grad_S_Fn=grad_S_Fn,
                                           H_kinetic_Fn=H_kinetic_Fn,
                                           eps=eps,
                                           N_steps=N_steps,
                                           record_H=False)
        # backward integration
        mom_b, phi_b = leapfrog_integrator(-mom_f,
                                           phi_f,
                                           S_Fn=S_Fn,
                                           grad_S_Fn=grad_S_Fn,
                                           H_kinetic_Fn=H_kinetic_Fn,
                                           eps=eps,
                                           N_steps=N_steps,
                                           record_H=False)
        # expect to recover initial conditions
        self.assertAlmostEqual(phi_0, phi_b, places=10)
        self.assertAlmostEqual(mom_0, -mom_b, places=10)

    def test_leapfrog_energy_conservation(self):
        omega = random_float_uniform(n=1, lower=-5.0, upper=5.0)[0]
        S_Fn, grad_S_Fn, H_kinetic_Fn = make_harmonic_energy_fns(omega=omega)
        phi_0 = random_float_uniform(n=1, lower=-10.0, upper=10.0)[0]
        mom_0 = random_float_uniform(n=1, lower=-10.0, upper=10.0)[0]
        eps = random_float_uniform(n=1, lower=0.01, upper=0.1)[0]
        N_steps = random_int_uniform(n=1, lower=10, upper=50)[0]
        # run integrator with energy recording
        mom_f, phi_f, H_hist = leapfrog_integrator(mom_0,
                                                   phi_0,
                                                   S_Fn=S_Fn,
                                                   grad_S_Fn=grad_S_Fn,
                                                   H_kinetic_Fn=H_kinetic_Fn,
                                                   eps=eps,
                                                   N_steps=N_steps,
                                                   record_H=True)
        # check that energy is approximately conserved
        H_initial = H_hist[0]
        max_dev = float(jnp.max(jnp.abs(H_hist - H_initial)))
        rel_dev = max_dev / float(jnp.abs(H_initial))
        self.assertLessEqual(rel_dev, 10*eps**2)


if __name__ == '__main__':
    unittest.main()
