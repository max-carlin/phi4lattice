import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from .energetics import action_core, grad_action_core
from .energetics import hamiltonian_kinetic_core
"""
Integrators methods for monte carlo.
"""


@staticmethod
# N_steps through record_H static
@partial(jax.jit, static_argnums=tuple(range(2, 11)))
def omelyan_core_scan(mom_x0, phi_x0,
                      N_steps,
                      lam, kappa, D,
                      shift, spatial_axes,
                      eps,
                      xi,
                      record_H):
    """
    One Omelyan trajectory of N_steps.
    If record_H -> also return H history (shape (N_steps+1, batch))
    """
    # pre-compute initial energy if Hamiltonian history is desired
    if record_H:
        S0, _, _ = action_core(phi_x0,
                               lam, kappa,
                               D, shift, spatial_axes)
        H0 = hamiltonian_kinetic_core(mom_x0, spatial_axes) + S0

    def om_step(state: tuple[jnp.ndarray, jnp.ndarray], _):
        # Scan expects an x input along with carry/state even though xs=none
        mom_x_p, phi_x_p = state

        # I1(ξ eps)
        phi_x_p = phi_x_p + eps * xi*mom_x_p
        # I2(eps/2)
        grad_s = grad_action_core(phi_x_p, lam,
                                  kappa, D,
                                  shift, spatial_axes)
        # grad_s = Phi4Lattice._grad_action_core(phi_x_p, lam, kappa, D)
        mom_x_p = mom_x_p - eps / 2 * grad_s
        # I1((1-2ξ)eps)
        phi_x_p = phi_x_p + ((1-2*xi)*eps)*mom_x_p
        # I2(eps/2)
        grad_s_p = grad_action_core(phi_x_p, lam,
                                    kappa, D,
                                    shift, spatial_axes)
        # grad_s_p = Phi4Lattice._grad_action_core(phi_x_p, lam, kappa, D)
        mom_x_p = mom_x_p - eps / 2 * grad_s_p
        # I1(ξ eps)
        phi_x_p = phi_x_p + eps*xi*mom_x_p

        if record_H:
            S_p, _, _ = action_core(phi_x_p, lam,
                                    kappa, D,
                                    shift, spatial_axes)
            H_p = hamiltonian_kinetic_core(mom_x_p, spatial_axes) + S_p
            return (mom_x_p, phi_x_p), H_p
        return (mom_x_p, phi_x_p), None

    # run om_step for N_steps; xs not needed (output only depends on state)
    # H_hist is the ys -- array of what the Hi values were after each step
    (mom_fx, phi_fx), H_hist = lax.scan(om_step,
                                        (mom_x0, phi_x0),
                                        xs=None,
                                        length=N_steps)

    if record_H:
        # include initial H val at position 0
        H_hist = jnp.concatenate((H0[None], H_hist), axis=0)
        return mom_fx, phi_fx, H_hist
    return mom_fx, phi_fx


@staticmethod
@partial(jax.jit, static_argnums=range(2, 10))  # static: N_steps...spatial
def leapfrog_core_scan(mom_x0: jnp.ndarray,
                       phi_x0: jnp.ndarray,
                       eps: float,
                       N_steps: int,
                       lam: float, kappa: float,
                       D: int, shift: int, spatial_axes: tuple[int, ...],
                       record_H: bool):
    '''Run the leapfrog integrator for N_steps using JAX lax.scan.

    Parameters
    ----------
    mom_x0 : jnp.ndarray
        Initial momentum field.
    phi_x0 : jnp.ndarray
        Initial field configuration.
    params : HMCParams
        Simulation parameters

    Returns
    -------
    mom_fx, phi_fx : tuple
    '''

    eps = eps
    N_steps = N_steps
    lam = lam
    kappa = kappa
    D = D
    shift = shift
    spatial_axes = spatial_axes
    record_H = record_H

    # compute initial H if history is desired
    if record_H:
        S0, _, _ = action_core(phi_x0, lam, kappa, D, shift, spatial_axes)
        H0 = hamiltonian_kinetic_core(mom_x0, spatial_axes) + S0

    def leap_step(state: tuple[jnp.ndarray, jnp.ndarray], _):
        '''
        Perform a single step of the leapfrog
        integration for the lattice field.

        Parameters
        ----------
        state : tuple
            Current system state (mom_x_p, phi_x_p)
        _ : any
            Unused placeholder for lax.scan compatability.
        params : HMCparams
            Simulation parameters and lattice geometry.

        Returns
        -------
        (mom_x_p, phi_x_p), H_p : tuple
            mom_x_p, phi_x_p are the updated momentum and field.
            H_p is the Hamiltonian if 'record_H' is True, else None.
        '''

        mom_x_p, phi_x_p = state

        # I1 first half step; phi updates
        phi_x_p = phi_x_p + eps/2 * mom_x_p

        # I2 pi updates whole step
        grad_s = grad_action_core(phi_x_p, lam,
                                  kappa, D,
                                  shift, spatial_axes)

        # Grad_s = Phi4Lattice._grad_action_core(phi_x_p, lam, kappa, D)
        mom_x_p = mom_x_p - eps*grad_s

        # I1 second half step; phi updates again
        phi_x_p = phi_x_p + eps/2 * mom_x_p

        # Compute updated H after step
        if record_H:
            S_p, _, _ = action_core(phi_x_p, lam, kappa,
                                    D, shift, spatial_axes)
            H_p = hamiltonian_kinetic_core(mom_x_p, spatial_axes) + S_p

            return (mom_x_p, phi_x_p), H_p
        return (mom_x_p, phi_x_p), None

    # run leap_step for N_steps
    (mom_fx, phi_fx), H_hist = lax.scan(lambda s, _: leap_step(s, _, params),
                                        (mom_x0, phi_x0), xs=None,
                                        length=N_steps)

    if record_H:
        H_hist = jnp.concatenate((H0[None], H_hist), axis=0)
        return mom_fx, phi_fx, H_hist
    return mom_fx, phi_fx
