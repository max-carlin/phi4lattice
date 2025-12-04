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
from jax import lax
import integrators as integ
from params import HMCConfig
from typing import Callable
from functools import partial
jax.config.update("jax_enable_x64", True)


def HMC_core(H_old: jnp.ndarray, H_prime: jnp.ndarray,
             phi_old: jnp.ndarray, phi_prime: jnp.ndarray,
             mom_old: jnp.ndarray, mom_prime: jnp.ndarray,
             key: jnp.ndarray
             ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    # these might make jit fail so we should test outside
    # of jit if possible.
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


def MD_traj(state: tuple[jnp.ndarray, jnp.ndarray],
            traj_key_pair: tuple[jnp.ndarray, jnp.ndarray],
            *,
            cfg: HMCConfig,
            S_Fn: Callable,
            grad_S_Fn: Callable,
            H_kinetic_Fn: Callable,
            measure_fns_dict: dict[str, Callable] = None
            ):
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
    cfg : HMCConfig
        Holds all integration and lattice geometry settings.
    measure_fns_dict : dict, optional
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
    mom_key, r_key = traj_key_pair
    # validate inputs
    _validate_md_traj_inputs(state, traj_key_pair, cfg)
    traj_out_dict: dict = {}

    # 1) refresh momentum field at the start of each trajectory
    mom_refreshed = random.normal(mom_key,
                                  shape=mom_old.shape,
                                  dtype=mom_old.dtype)

    # 2) run MD integrator to get proposed (mom_fx, phi_fx)
    if cfg.integrator == 'omelyan':
        result = integ.omelyan_integrator(mom_refreshed,
                                          phi_old,
                                          S_Fn=S_Fn,
                                          grad_S_Fn=grad_S_Fn,
                                          H_kinetic_Fn=H_kinetic_Fn,
                                          eps=cfg.eps,
                                          xi=cfg.xi,
                                          N_steps=cfg.N_steps,
                                          record_H=cfg.record_H)
    elif cfg.integrator == 'leapfrog':
        result = integ.leapfrog_integrator(mom_refreshed,
                                           phi_old,
                                           S_Fn=S_Fn,
                                           grad_S_Fn=grad_S_Fn,
                                           H_kinetic_Fn=H_kinetic_Fn,
                                           eps=cfg.eps,
                                           N_steps=cfg.N_steps,
                                           record_H=cfg.record_H)
    else:
        raise ValueError(f"Unknown integrator {cfg.integrator}. "
                         "Expected 'omelyan' or 'leapfrog'")

    if cfg.record_H:
        mom_fx, phi_fx, H_hist = result
        traj_out_dict['H_hist'] = H_hist
    else:
        mom_fx, phi_fx = result

    # 3) Metropolis accept/reject step
    if cfg.metropolis:
        H_old = H_kinetic_Fn(mom_refreshed) + S_Fn(phi_old)
        H_prime = H_kinetic_Fn(mom_fx) + S_Fn(phi_fx)

        mom_acc, phi_acc, accept_mask, delta_H = HMC_core(H_old,
                                                          H_prime,
                                                          phi_old,
                                                          phi_fx,
                                                          mom_refreshed,
                                                          mom_fx,
                                                          r_key)
        # overwrite final fields if accepted
        mom_fx, phi_fx = mom_acc, phi_acc
        traj_out_dict['traj_mom_key'] = mom_key
        traj_out_dict['traj_metropolis_key'] = r_key
        traj_out_dict['accept_mask'] = accept_mask
        traj_out_dict['delta_H'] = delta_H

    # 4) measurements on final field if desired
    if measure_fns_dict:
        for name, fn in measure_fns_dict.items():
            traj_out_dict[name] = fn(phi_fx)

    return (mom_fx, phi_fx), traj_out_dict


def _validate_md_traj_inputs(state,
                             traj_key_pair,
                             cfg):
    mom_old, phi_old = state
    mom_key, r_key = traj_key_pair

    if mom_old.shape != phi_old.shape:
        raise ValueError("Momentum and field have different shapes.")
    if not isinstance(cfg, HMCConfig):
        raise ValueError("cfg must be an instance of HMCConfig.")
    if mom_key.shape != (2,) or r_key.shape != (2,):
        raise ValueError("Each key in traj_key_pair must have shape (2,).")


# @jax.jit(static_argnames=("cfg", "S_Fn", "grad_S_Fn",
#                           "H_kinetic_Fn", "measure_fns_dict"))
# @partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def run_HMC_trajectories(phi0: jnp.ndarray,
                         mom0: jnp.ndarray,
                         traj_keys: jnp.ndarray,  # shape (N_traj, 2, 2)
                         cfg: HMCConfig,
                         S_Fn: Callable,
                         grad_S_Fn: Callable,
                         H_kinetic_Fn: Callable,
                         measure_fns_dict: dict[str, Callable] = None
                         ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], dict]:
    """
    Wrapper for run_HMC_trajectories_core to allow for JIT compilation.
    Accepts measure_fns_dict as a dictionary (not hashable for JAX JIT) and
    converts to tuple of items and passes to jitted core function with
    hashable representation of tuple, from which the dict can be reconstructed.

    old error:
        ValueError: Non-hashable static arguments are not supported.
                    An error occurred while trying to hash an object
                    of type <class 'dict'>,
                    {'magnetization': <function magnetization at 0x12d5e5940>}.
                    The error was:
        TypeError: unhashable type: 'dict'
    """
    if measure_fns_dict is not None:
        if not isinstance(measure_fns_dict, dict):
            raise ValueError("measure_fns_dict must be a dictionary "
                             "of name:function pairs.")
        for name, fn in measure_fns_dict.items():
            if not callable(fn):
                raise ValueError(f"measure_fns_dict[{name}] is not callable.")

        measure_fns_items = tuple(measure_fns_dict.items())

    else:
        measure_fns_items = measure_fns_dict

    return run_HMC_trajectories_core(phi0,
                                     mom0,
                                     traj_keys,
                                     cfg,
                                     S_Fn,
                                     grad_S_Fn,
                                     H_kinetic_Fn,
                                     measure_fns_items=measure_fns_items
                                     )


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def run_HMC_trajectories_core(phi0: jnp.ndarray,
                              mom0: jnp.ndarray,
                              traj_keys: jnp.ndarray,  # shape (N_traj, 2, 2)
                              cfg: HMCConfig,
                              S_Fn: Callable,
                              grad_S_Fn: Callable,
                              H_kinetic_Fn: Callable,
                              measure_fns_items: tuple[tuple[str,
                                                             Callable]] = None
                              ) -> tuple[tuple[jnp.ndarray,
                                               jnp.ndarray], dict]:
    """
    Wrap MD_traj to run multiple HMC trajectories using JAX lax.scan.
    """
    if measure_fns_items is not None:
        measure_fns_dict = dict(measure_fns_items)
    else:
        measure_fns_dict = None

    if traj_keys.shape != (cfg.N_trajectories, 2, 2):
        raise ValueError("traj_keys must have shape "
                         f"({cfg.N_trajectories}, 2, 2); "
                         f"got {traj_keys.shape}.")

    def one_traj(state, traj_key_pair):
        (mom_x, phi_x) = state
        (mom_fx, phi_fx), out = MD_traj((mom_x, phi_x),
                                        traj_key_pair,
                                        cfg=cfg,
                                        S_Fn=S_Fn,
                                        grad_S_Fn=grad_S_Fn,
                                        H_kinetic_Fn=H_kinetic_Fn,
                                        measure_fns_dict=measure_fns_dict)
        return (mom_fx, phi_fx), out

    (mom_final, phi_final), traj_outs_dict = lax.scan(
                                                    one_traj,
                                                    (mom0, phi0),
                                                    xs=traj_keys,
                                                    length=cfg.N_trajectories
                                                    )
    return (mom_final, phi_final), traj_outs_dict
