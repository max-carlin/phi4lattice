"""
Integrator methods for Hybrid Monte Carlo (HMC).

This module provides low-level numerical integrators used to evolve the
field and momentum system during a Molecular Dynamics (MD) trajectory.
It is designed to be interacted with through either hmc.py's
run_HMC_trajectory or MD_traj functions. Both integrators optionally
record the Hamiltonian at each step.

It contains the following custom functions:
    * omelyan_integrator :
        Runs one step of omelyan numerical integration.
    * leapfrog_integrator :
        Runs one step of leapfrog numerical integration.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Callable
jax.config.update("jax_enable_x64", True)


def omelyan_integrator(mom_x0: jnp.ndarray,
                       phi_x0: jnp.ndarray,
                       *,  # required named args
                       S_Fn: Callable,
                       grad_S_Fn: Callable,
                       H_kinetic_Fn: Callable,
                       eps: float,
                       xi: float,
                       N_steps: int,
                       record_H: bool):
    """One Omelyan trajectory of N_steps.

    This function performs one step of the Omelyan
    numerical integration scheme. It will also optionally
    return H history.

    Parameters
    ----------
    mom_x0 : jnp.ndarray
        Initial momentum field.
    phi_x0 : jnp.ndarray
        Initial field configuration.
    S_Fn : Callable
        Function computing the action.
    grad_S_Fn : Callable
        Function computing gradient.
    H_kinetic_Fn : Callable
        Function computing the kinetic term.
    eps : float
        Integrator step size.
    xi : float
        Omelyan parameter.
    N_steps : int
        Number of MD integration steps.
    record_H : bool
        Whether to record the Hamiltonian values at each step.

    Returns
    -------
    mom_fx : jnp.ndarray
        Final momentum field after integration.
    phi_fx : jnp.ndarray
        Final field after integration.
    H_hist : jnp.ndarray, optional
        Hamiltonian history if `record_H` is True.


    Notes
    -----
    This is a low-level numerical kernel. It assumes inputs have been
    validated (e.g. via HMCConfig and MD_traj) and performs minimal
    error checking. For typical use, call it indirectly through
    MD_traj / run_HMC_trajectories.
    """
    # pre-compute initial energy if Hamiltonian history is desired
    if record_H:
        S0 = S_Fn(phi_x0)
        H0 = H_kinetic_Fn(mom_x0) + S0

    def om_step(state: tuple[jnp.ndarray, jnp.ndarray], _):
        # Scan expects an x input along with carry/state even though xs=none
        mom_x_p, phi_x_p = state

        # I1(ξ eps)
        phi_x_p = phi_x_p + eps * xi*mom_x_p
        # I2(eps/2)
        mom_x_p = mom_x_p - eps / 2 * grad_S_Fn(phi_x_p)
        # I1((1-2ξ)eps)
        phi_x_p = phi_x_p + ((1-2*xi)*eps)*mom_x_p
        # I2(eps/2)
        mom_x_p = mom_x_p - eps / 2 * grad_S_Fn(phi_x_p)
        # I1(ξ eps)
        phi_x_p = phi_x_p + eps*xi*mom_x_p

        if record_H:
            S_p = S_Fn(phi_x_p)
            H_p = H_kinetic_Fn(mom_x_p) + S_p
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


def leapfrog_integrator(mom_x0: jnp.ndarray,
                        phi_x0: jnp.ndarray,
                        *,  # required named args
                        S_Fn: Callable,
                        grad_S_Fn: Callable,
                        H_kinetic_Fn: Callable,
                        eps: float,
                        N_steps: int,
                        record_H: bool):
    '''
    Run the leapfrog integrator for N_steps using JAX lax.scan.

    Parameters
    ----------
    mom_x0 : jnp.ndarray
        Initial momentum field.
    phi_x0 : jnp.ndarray
        Initial field configuration.
    S_Fn : Callable
        Function computing the action.
    grad_S_Fn : Callable
        Function computing the gradient.
    H_kinetic_Fn : Callable
        Function computing the kinetic term.
    eps : float
        Integrator step size.
    N_steps : int
        Number of MD integration steps.
    record_H : bool
        Whether to record Hamiltonian values at each step.

    Returns
    -------
    mom_fx : jnp.ndarray
        Final momentum field.
    phi_fx : jnp.ndarray
        Final field after integration.
    H_hist : jnp.ndarray, optional
        Hamiltonian history of shape (N_steps + 1,) if `record_H` is True.

    Notes
    -----
    This is a low-level numerical kernel. It assumes inputs have been
    validated (e.g. via HMCConfig and MD_traj) and performs minimal
    error checking. For typical use, call it indirectly through
    MD_traj / run_HMC_trajectories.
    '''

    # compute initial H if history is desired
    if record_H:
        S0 = S_Fn(phi_x0)
        H0 = H_kinetic_Fn(mom_x0) + S0

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

        mom_x_p = mom_x_p - eps*grad_S_Fn(phi_x_p)

        # I1 second half step; phi updates again
        phi_x_p = phi_x_p + eps/2 * mom_x_p

        # Compute updated H after step
        if record_H:
            S_p = S_Fn(phi_x_p)
            H_p = H_kinetic_Fn(mom_x_p) + S_p

            return (mom_x_p, phi_x_p), H_p
        return (mom_x_p, phi_x_p), None

    # run leap_step for N_steps
    (mom_fx, phi_fx), H_hist = lax.scan(leap_step,
                                        (mom_x0, phi_x0),
                                        xs=None,
                                        length=N_steps)

    if record_H:
        H_hist = jnp.concatenate((H0[None], H_hist), axis=0)
        return mom_fx, phi_fx, H_hist
    return mom_fx, phi_fx
