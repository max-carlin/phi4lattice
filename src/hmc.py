'''Molecular Dynamics trajectory and acceptance for Hybrid Monte Carlo.

This script contains the core functions used to advance the scalar field
through one HMC trajectory and to decide whether to accept the resulting
update.

It contains the following custom functions:
    * HMC_core :
        Determines whether to accept or reject a field based
        on the energy differences.
    * MD_traj
        Runs one full molecular dynamics trajectory by refreshing the
        momentum field, evolving the system with a numerical integrator,
        and optionally performing an acceptance test.
'''
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as random
from .action import action_core
from .integrators import omelyan_core_scan, leapfrog_core_scan
from .integrators import hamiltonian_kinetic_core
from .prng import make_keys, randomize_normal_core
from .params import HMCParams


@staticmethod
@jax.jit
def HMC_core(H_old, H_prime,
             phi_old, phi_prime,
             mom_old, mom_prime,
             key):
    '''Accept or reject a proposed HMC trajectory.

    The function computes the energy difference
    Delta H = H_{prime} - H_{old} and accepts the proposed state
    when Delta H < 0 or when the random probability is less than exp(-Delta H).
    The random number is drawn from a uniform distribution using 'key'.

    Parameters
    ----------
    H_old :  jnp.ndarray
        Hamiltonian before run through MD_traj.
    H_prime : jnp.ndarray
        Hamiltonian after the update. Same shap as H_old.
    phi_old : jnp.ndarray
        Current field configuration.
    phi_prime : jnp.ndarry
        Field configuration after the update.
    mom_old : jnp.ndarray
        Current omoentum field.
    mom_prime : jnp.ndarray
        Momentum field after the update.

    Returns
    -------
    mom_accepted : jnp.ndarray
        Momentum field after applying accept / reject.
    phi_accepted : jnp.ndarray
        Field after applying accept / reject.
    accept_mask : jnp.ndarray
        Boolean array with what values were accepted.
    delta_H : jnp.ndarray
        The energy difference.
    '''
    # First, we want to catch possible ValueErrors
    if H_old.shape != H_prime.shape:
        raise ValueError("Hamiltonians have different shapes.")
    if phi_old.shape != phi_prime.shape:
        raise ValueError("Fields have different shapes.")
    if mom_old.shape != mom_prime.shape:
        raise ValueError("Momentum fields have different shapes.")

    delta_H = H_prime - H_old  # H_final - H_initial
    # make acceptor mask
    r = random.uniform(key, shape=delta_H.shape)
    # accept if H_prime < H_old or if r < exp(-delta_H)
    accept_mask = (delta_H < 0) | (r < jnp.exp(-delta_H))
    # reshape mask for batched
    #   will throw value error when batched if I don't
    #     ex) ValueError: Incompatible shapes for broadcasting:
    #         shapes=[(10,), (10, 4, 4, 4, 4), (10, 4, 4, 4, 4)]
    #   for batch of 10 configs
    #   accept_mask.ndim = 0 for non batched, it's just a scalar
    #   mask = accept_mask
    # accept_mask.ndim = 1 for batched (basically the same as self.shift)
    # should give shape of (10, 1,1,1,1) for the above example
    mask = accept_mask.reshape(accept_mask.shape
                               + (1,) * (phi_old.ndim - accept_mask.ndim))
    phi_accepted = jnp.where(mask, phi_prime, phi_old)
    mom_accepted = jnp.where(mask, mom_prime, mom_old)
    return mom_accepted, phi_accepted, mask, delta_H


def MD_traj(state,
            key_pair,
            params: HMCParams,
            measure_fns=None):
    '''Run one molecular dynamics (MD) trajectory step.

    This function takes the current field and momentum values, refreshes
    the momentum from a normal distribution, and evolves the system forward
    with a numerical (leapfrog or omelyan) integrator. Optionally applies
    HMC_core to accept / reject the new state.

    Parameters
    ----------
    state : tuple
        The current (momentum, field) state of the system.
    key_pair : tuple
        Random number generator keys.
    params : HMCParams
        Holds all integration and lattice geometry settings.
    measure_fns : dict, optional
        Functions to gather measurements on the final field.

    Returns
    -------
    (mom_fx, phi_fx) : tuple
        The updated momentum and field after one trajectory.
    out : dict
        Extra outputs such as acceptance info, Hamiltonian history, or
        measurement results.
    '''
    mom_old, phi_old = state
    # one key for momentum refresh, one for metropolis
    mom_key, r_key = key_pair
    out: dict = {}

    # Set up integrator params
    # 1) refresh momentum field at the start of each trajectory
    mom_master_key, mom_keys = make_keys(mom_old.shape[0], mom_key)
    mom_refreshed = randomize_normal_core(mom_keys,
                                          params.lat_shape,
                                          mu=0,
                                          sigma=1)

    # Placeholder so output is defined
    output = None

    # .lower() puts the string all in lower case
    if params.integrator[:7].lower() == 'omelyan':
        output = omelyan_core_scan(mom_refreshed, phi_old, params)
    # For these checks, any variation of leapfrog and omelyan work
    if params.integrator[:4].lower() == 'leap':
        output = leapfrog_core_scan(mom_refreshed, phi_old, params)

    if output is None:
        raise ValueError(
            f"Unknown integrator {params.integrator}. "
            "Expected 'omelyan' or 'leap'"
        )

    mom_fx = output[0]
    phi_fx = output[1]
    if params.record_H:
        out['H_hist'] = output[2]

    if params.metropolis:
        # 5) calc H_f
        mom_term_prime = hamiltonian_kinetic_core(mom_fx, params.spatial_axes)
        S_prime, _, _ = action_core(phi_fx,
                                    params.lam,
                                    params.kappa,
                                    params.D,
                                    params.shift,
                                    params.spatial_axes)
        H_prime = mom_term_prime + S_prime

        mom_term_old = hamiltonian_kinetic_core(mom_refreshed,
                                                params.spatial_axes)
        S_old, _, _ = action_core(phi_old,
                                  params.lam,
                                  params.kappa,
                                  params.D,
                                  params.shift,
                                  params.spatial_axes)
        H_old = mom_term_old + S_old

        # 3) current H val
        # 6) Metropolis test to update fields after each trajectory (tau)
        mom_acc, phi_acc, accept_mask, delta_H = HMC_core(H_old, H_prime,
                                                          phi_old,
                                                          phi_fx,
                                                          mom_refreshed,
                                                          mom_fx,
                                                          r_key)
        mom_fx = mom_acc  # overwrite final fields if accepted
        phi_fx = phi_acc
        out['traj_mom_keys'] = mom_keys
        out['traj_r_keys'] = r_key
        out['accept_mask'] = accept_mask
        out['delta_H'] = delta_H

    if measure_fns:
        for name, fn in measure_fns.items():
            out[name] = fn(phi_fx)

    return (mom_fx, phi_fx), out
