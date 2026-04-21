import jax
import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax import lax
import integrators as integ
from params import MetropMCConfig
from typing import Callable
from functools import partial
jax.config.update("jax_enable_x64", True)


# def Metropolis(S_old: jnp.ndarray, S_new: jnp.ndarray,
#                sigma_old: jnp.ndarray, sigma_new: jnp.ndarray,
#                r_key: jnp.ndarray):
#     """Core Metropolis accept/reject logic.

#     Parameters
#     ----------
#     S_old : jnp.ndarray
#         Action of current configuration.
#     S_new : jnp.ndarray
#         Action of proposed new configuration.
#     sigma_old : jnp.ndarray
#         Current configuration.
#     sigma_new : jnp.ndarray
#         Proposed new configuration.
#     key : jnp.ndarray
#         PRNG key for generating random number for acceptance.

#     Returns
#     -------
#     accept : bool
#         Whether to accept the new configuration.
#     """
#     if S_old.shape != S_new.shape:
#         raise ValueError(f"S_old and S_new must have the same shape, "
#                          f"got {S_old.shape} and {S_new.shape}.")
#     if sigma_old.shape != sigma_new.shape:
#         raise ValueError(f"sigma_old and sigma_new must have the same shape, "
#                          f"got {sigma_old.shape} and {sigma_new.shape}.")

#     delta_S = S_new - S_old
#     r = random.uniform(r_key, shape=delta_S.shape)
#     # accept if new action is lower or equal
#     # or with probability exp(-delta_S) if higher
#     accept_mask = (delta_S <= 0) | (r < jnp.exp(-delta_S))

#     mask = accept_mask.reshape(accept_mask.shape
#                                + (1,) * (sigma_old.ndim - accept_mask.ndim))
#     sigma_accepted = jnp.where(mask, sigma_new, sigma_old)
#     return sigma_accepted, mask, delta_S

def Metropolis_delta_S(delta_S: jnp.ndarray,
                       sigma_old: jnp.ndarray,
                       sigma_new: jnp.ndarray,
                       r_key: jnp.ndarray):
    r = random.uniform(r_key, shape=delta_S.shape)
    accept_mask = (delta_S <= 0) | (r < jnp.exp(-delta_S))

    mask = accept_mask.reshape(accept_mask.shape
                               + (1,) * (sigma_old.ndim - accept_mask.ndim))
    sigma_accepted = jnp.where(mask, sigma_new, sigma_old)
    return sigma_accepted, mask, delta_S

def MC_step(sigma_x: jnp.ndarray,
            step_key_pair: tuple[jnp.ndarray, jnp.ndarray],
            *,  # required named args
            cfg: MetropMCConfig,
            S_Fn: Callable,
            propose_flip_Fn: Callable,
            local_delta_S_Fn: Callable,
            measure_fns_dict: dict[str, Callable] = None,
            store_proposal_history: bool = False):
    """
    Perform one Metropolis MC step.
    Perform one Metropolis MC step: pick one site, spin flip, accept/rejct.

    Parameters
    ----------
    sigma_x : jnp.ndarray
        Current field configuration.
    step_key_pair : tuple[jnp.ndarray, jnp.ndarray]
        PRNG keys for this step.
    cfg : MetropMCConfig
        Configuration parameters for the MC step.
    S_Fn : Callable
        Function to compute the action of a configuration.
    measure_fns_dict : dict[str, Callable], optional
        Dictionary of measurement functions to apply to the new configuration.

    Returns
    -------
    sigma_x_new : jnp.ndarray
        New field configuration after the MC step.
    accept : bool
        Whether the new configuration was accepted.
    measures : dict[str, float], optional
        Dictionary of measured observables if `measure_fns_dict` is provided.
    """
    out_dict = {}

    site_key, r_key = step_key_pair


    # propose new configuration by flipping sign of random site
    sigma_x_new, site_coords, old_spin = propose_flip_Fn(sigma_x,
                                               site_key)

    # compute local change in action from proposed flip
    delta_S = local_delta_S_Fn(sigma_x, site_coords)


    # accept or reject new configuration
    sigma_acc, accept_mask, delta_S = Metropolis_delta_S(delta_S,
                                                         sigma_x,
                                                         sigma_x_new,
                                                         r_key)
    if store_proposal_history:
        out_dict['traj_site_key'] = site_key
        out_dict['traj_r_key'] = r_key
        out_dict['site'] = site_coords
        out_dict['accept_mask'] = accept_mask
        out_dict['delta_S'] = delta_S

    if measure_fns_dict is not None:
        for name, fn in measure_fns_dict.items():
            out_dict[name] = fn(sigma_acc)

    return sigma_acc, out_dict


def run_Metropolis_MC(sigma_x0: jnp.ndarray,
                      sweep_keys: jnp.ndarray,
                      cfg: MetropMCConfig,
                      S_Fn: Callable,
                      propose_flip_Fn: Callable,
                      local_delta_S_Fn: Callable,
                      measure_fns_dict: dict[str, Callable] = None,
                      store_proposal_history: bool = False,
                      thermalization_summary: bool = False):
    """
    Run a Metropolis MC trajectory.

    Parameters
    ----------
    sigma_x0 : jnp.ndarray
        Initial field configuration.
    sweep_keys : jnp.ndarray
        PRNG keys for generating random numbers.
    cfg : MetropMCConfig
        Configuration parameters for the MC steps."""

    if measure_fns_dict is not None:
        if not isinstance(measure_fns_dict, dict):
            raise ValueError("measure_fns_dict must "
                             "be a dictionary if provided.")
        for name, fn in measure_fns_dict.items():
            if not callable(fn):
                raise ValueError(f"measure_fns_dict[{name}] is not callable.")
        measure_fns_items = tuple(measure_fns_dict.items())

    else:
        measure_fns_items = measure_fns_dict

    return run_Metropolis_MC_core(sigma_x0,
                                  sweep_keys,
                                  cfg,
                                  S_Fn,
                                  propose_flip_Fn,
                                  local_delta_S_Fn,
                                  measure_fns_items,
                                  store_proposal_history=store_proposal_history,
                                  thermalization_summary=thermalization_summary)


@partial(jax.jit, static_argnames=['cfg',
                                   'S_Fn',
                                   'propose_flip_Fn',
                                   'local_delta_S_Fn',
                                   'measure_fns_items',
                                   'store_proposal_history',
                                   'thermalization_summary'])
def run_Metropolis_MC_core(sigma_x0: jnp.ndarray,
                           sweep_keys: jnp.ndarray,
                           cfg: MetropMCConfig,
                           S_Fn: Callable,
                           propose_flip_Fn: Callable,
                           local_delta_S_Fn: Callable,
                           measure_fns_items: tuple[
                               tuple[str, Callable], ...] = None,
                           store_proposal_history: bool = False,
                           thermalization_summary: bool = False):
    """
    Run a Metropolis MC trajectory.

    Parameters
    ----------
    sigma_x0 : jnp.ndarray
        Initial field configuration.
    sweep_keys : jnp.ndarray
        PRNG keys for generating random numbers.
    N_steps : int
        Number of MC steps to run.
    cfg : MetropMCConfig
        Configuration parameters for the MC steps."""
    if measure_fns_items is not None:
        measure_fns_dict = dict(measure_fns_items)
    else:
        # make a copy to avoid mutating input
        measure_fns_dict = None

    if sweep_keys.shape != (cfg.N_steps, 2, 2):
        raise ValueError(f"sweep_keys must have shape "
                         f"({cfg.N_steps}, 2, 2), "
                         f"got {sweep_keys.shape}.")

    # S_0 = S_Fn(sigma_x0)
    # init_carry = {"S_current": S_0,
    #               "sigma_x": sigma_x0,
    #               "sum_S": 0.0,
    #               "sum_S2": 0.0}

    
    def one_step_thermalization(carry, step_key_pair):
        sigma_x = carry["sigma_x"]
        S_current = carry["S_current"]
        sum_S = carry["sum_S"]
        sum_S2 = carry["sum_S2"]
        M_current = carry["M_current"]
        sum_abs_M = carry["sum_abs_M"]
        sum_abs_M2 = carry["sum_abs_M2"]

        site_key, r_key = step_key_pair

        sigma_prop, site_coords, old_spin = propose_flip_Fn(sigma_x,
                                                  site_key)
        delta_S = local_delta_S_Fn(sigma_x, site_coords)
        sigma_acc, accept_mask, delta_S = Metropolis_delta_S(delta_S,
                                                                sigma_x,
                                                                sigma_prop,
                                                                r_key)
        
        accepted = accept_mask.reshape(delta_S.shape)
        S_new = jnp.where(accepted, S_current + delta_S, S_current)
        sum_S += S_new
        sum_S2 += S_new**2

        # deltaM = sig_i'-sig_i = (-sig_i) - sig_i = -2*sig_i
        delta_M = -2 * old_spin
        M_new = jnp.where(accepted, M_current + delta_M, M_current)
        abs_M_new = jnp.abs(M_new)
        sum_abs_M += abs_M_new
        sum_abs_M2 += abs_M_new**2


        carry = {"sigma_x": sigma_acc,
                 "S_current": S_new,
                 "sum_S": sum_S,
                 "sum_S2": sum_S2,
                 "M_current": M_new,
                 "sum_abs_M": sum_abs_M,
                 "sum_abs_M2": sum_abs_M2}
        return carry, None
    
    if thermalization_summary:
        S_0 = S_Fn(sigma_x0)
        zero_S = jnp.zeros_like(S_0)
        if S_0.shape == ():
            M_0 = sigma_x0.sum()
            zero_M = jnp.zeros_like(M_0)
        else:
            M_0 = sigma_x0.sum(axis=tuple(range(1, sigma_x0.ndim)))
            zero_M = jnp.zeros_like(M_0)


        init_carry = {"S_current": S_0,
                      "sigma_x": sigma_x0,
                      "sum_S": zero_S,
                      "sum_S2": zero_S,
                      "M_current": M_0,
                      "sum_abs_M": zero_M,
                      "sum_abs_M2": zero_M}
        final_carry, _ = lax.scan(one_step_thermalization,
                                          init_carry,
                                          xs=sweep_keys,
                                          length=cfg.N_steps)
        action_mean = final_carry["sum_S"] / cfg.N_steps
        action_var = (final_carry["sum_S2"] / cfg.N_steps) - action_mean**2
        # floating point errors can make action_var slightly negative,
        # so take max with 0 before sqrt
        action_std = jnp.sqrt(jnp.maximum(action_var, 1e-8))

        abs_mag_mean = final_carry["sum_abs_M"] / cfg.N_steps
        abs_mag_var = (final_carry["sum_abs_M2"] / cfg.N_steps) - abs_mag_mean**2
        abs_mag_std = jnp.sqrt(jnp.maximum(abs_mag_var, 1e-8))

        summary_dict = {"action_mean": action_mean,
                   "action_std": action_std,
                   "abs_mag_mean": abs_mag_mean,
                   "abs_mag_std": abs_mag_std
                   }
        return final_carry["sigma_x"], summary_dict

    def one_step(sigma_x, step_key_pair):
        sigma_x_new, out_dict = MC_step(sigma_x,
                                        step_key_pair,
                                        cfg=cfg,
                                        S_Fn=S_Fn,
                                        propose_flip_Fn=propose_flip_Fn,
                                        measure_fns_dict=measure_fns_dict,
                                        local_delta_S_Fn=local_delta_S_Fn,
                                        store_proposal_history=store_proposal_history)
        return sigma_x_new, out_dict

    sigma_final, out_dicts = lax.scan(one_step,
                                      sigma_x0,
                                      xs=sweep_keys,
                                      length=cfg.N_steps)
    return sigma_final, out_dicts


    # def one__step(carry, step_key_pair):
    #     sigma_x = carry["sigma_x"]
    #     S_current = carry["S_current"]
    #     site_key, r_key = step_key_pair
        
    #     sigma_prop, site_coords = propose_flip_Fn(sigma_x,
    #                                               site_key)
    #     delta_S = local_delta_S_Fn(sigma_x, site_coords)

    #     sigma_acc, accept_mask, delta_S = Metropolis_delta_S(delta_S,
    #                                                          sigma_x,
    #                                                          sigma_prop,
    #                                                          r_key)
    #     accepted = accept_mask.reshape(delta_S.shape)
    #     # update sigma_x based on acceptance
    #     S_new = jnp.where(accepted,
    #                       S_current + delta_S,
    #                       S_current)
        
    #     sum_S += S_new
    #     sum_S2 += S_new**2

    #     carry = {"sigma_x": sigma_acc,
    #              "S_current": S_new,
    #              "sum_S": sum_S,
    #              "sum_S2": sum_S2}
    #     return carry, 

