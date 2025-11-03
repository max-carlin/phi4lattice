import jax
from jax import lax
import numpy as np
import jax.numpy as jnp
import jax.random as random
from .action import action_core
from .integrators import omelyan_core_scan, leapfrog_core_scan
from .integrators import hamiltonian_kinetic_core
from .prng import make_keys, randomize_core
from .params import HMCParams


@staticmethod
@jax.jit
def HMC_core(H_old, H_prime,
             phi_old, phi_prime,
             mom_old, mom_prime,
             key):
    '''mask and update'''

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
    mom_old, phi_old = state
    # one key for momentum refresh, one for metropolis
    mom_key, r_key = key_pair
    out: dict = {}

    # Set up integrator params
    # 1) refresh momentum field at the start of each trajectory
    mom_master_key, mom_keys = make_keys(mom_old.shape[0], mom_key)
    mom_refreshed = randomize_core(mom_keys,
                                   params.lat_shape,
                                   mu=0,
                                   sigma=1)

    if params.integrator == 'omelyan':
        print("INTEGRATING")
        output = omelyan_core_scan(mom_refreshed, phi_old, params)
    if params.integrator == 'leap':
        output = leapfrog_core_scan(mom_refreshed, phi_old, params)

    print('OUTPUT', output)

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
