import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from .action import grad_action, action_core, grad_action_core
from .params import HMCParams


@staticmethod
@partial(jax.jit, static_argnums=1)
def hamiltonian_kinetic_core(mom_x, spatial_axes):
    #1/2∑_x p_x²
    return(0.5*(mom_x**2).sum(axis = spatial_axes))

def hamiltonian(self):
    S, K, W = self.action_kinetic_W()
    mom_term = hamiltonian_kinetic_core(self.mom_x, self.spatial_axes)
    return mom_term + S

def om_step(state, _, params): #scan expects an x input along with carry/state even though xs=none
    mom_x_p,phi_x_p = state

    #I1(ξ eps)
    phi_x_p = phi_x_p + params.eps * params.xi*mom_x_p
    #I2(eps/2)
    grad_s = grad_action_core(phi_x_p, params.lam, params.kappa, params.D, params.shift, params.spatial_axes)
    # grad_s = Phi4Lattice._grad_action_core(phi_x_p, lam, kappa, D)
    mom_x_p = mom_x_p - params.eps/2 *grad_s
    #I1((1-2ξ)eps)
    phi_x_p = phi_x_p + ((1-2*params.xi)*params.eps)*mom_x_p
    #I2(eps/2)
    grad_s_p = grad_action_core(phi_x_p, params.lam, params.kappa, params.D, params.shift, params.spatial_axes)
    # grad_s_p = Phi4Lattice._grad_action_core(phi_x_p, lam, kappa, D)
    mom_x_p = mom_x_p - params.eps/2 *grad_s_p
    #I1(ξ eps)
    phi_x_p = phi_x_p + params.eps*params.xi*mom_x_p

    if params.record_H:
        S_p, _, _ = action_core(phi_x_p, params.lam, params.kappa, params.D, params.shift, params.spatial_axes)
        H_p = hamiltonian_kinetic_core(mom_x_p, params.spatial_axes) + S_p
        return (mom_x_p, phi_x_p), H_p
    return (mom_x_p, phi_x_p), None



@staticmethod
@partial(jax.jit,
          static_argnums=(2,))  # N_steps through record_H static
def omelyan_core_scan(mom_x0, phi_x0, params: HMCParams):
    """
    One Omelyan trajectory of N_steps.
    If record_H -> also return H history (shape (N_steps+1, batch))
    """
    eps = params.eps
    N_steps = params.N_steps
    lam = params.lam
    kappa = params.kappa
    D = params.D
    shift = params.shift
    spatial_axes = params.spatial_axes
    record_H = params.record_H
    # pre-compute initial energy if Hamiltonian history is desired
    if record_H:
      S0, _, _ = action_core(phi_x0, lam, kappa, D, shift, spatial_axes)
      H0 = hamiltonian_kinetic_core(mom_x0, spatial_axes) + S0
    
    (mom_fx, phi_fx), H_hist = lax.scan(
        lambda s, _: om_step(s, _, params),
        (mom_x0, phi_x0),
        xs=None,
        length=params.N_steps,
    )

    if record_H:
      # include initial H val at position 0
      H_hist = jnp.concatenate((H0[None], H_hist), axis=0)
      return mom_fx, phi_fx, H_hist
    return mom_fx, phi_fx


def leap_step(state, _, params):
	mom_x_p,phi_x_p = state

	#I1 first half step; phi updates
	phi_x_p = phi_x_p + params.eps/2 * mom_x_p
	#I2 pi updates whole step
	grad_s = grad_action_core(phi_x_p, params.lam, params.kappa, params.D, params.shift, params.spatial_axes)
	# grad_s = Phi4Lattice._grad_action_core(phi_x_p, lam, kappa, D)
	mom_x_p = mom_x_p - params.eps*grad_s
	#I1 second half step; phi updates again
	phi_x_p = phi_x_p + params.eps/2 * mom_x_p

	# compute updated H after step
	if params.record_H:
		S_p, _,_ = action_core(phi_x_p, params.lam, params.kappa, params.D, params.shift, params.spatial_axes)
		H_p = hamiltonian_kinetic_core(mom_x_p, params.spatial_axes) + S_p

		return (mom_x_p, phi_x_p), H_p
	return (mom_x_p, phi_x_p), None



@staticmethod
@partial(jax.jit, static_argnums=range(2, 10)) #static: N_steps...spatial
# @partial(jax.jit, static_argnums=(3, 4, 6,7,8,9)) #static: N_steps...spatial
# @partial(jax.jit, static_argnums=range(4, 10)) #static: lam...spatial
def leapfrog_core_scan(mom_x0, phi_x0,  params: HMCParams):
	eps = params.eps
	N_steps = params.N_steps
	lam = params.lam
	kappa = params.kappa
	D = params.D
	shift = params.shift
	spatial_axes = params.spatial_axes
	record_H = params.record_H

                        
	# compute initial H if history is desired
	if record_H:
		S0, _, _ = action_core(phi_x0, lam, kappa, D, shift, spatial_axes)
		H0 = hamiltonian_kinetic_core(mom_x0, spatial_axes) + S0

	#run leap_step for N_steps     
	(mom_fx, phi_fx), H_hist = lax.scan(
          lambda s, _: leap_step(s, _, params),
            (mom_x0, phi_x0),xs=None, length=params.N_steps)
     
	# Infer length from shape of xs so that I can vmap with dynamic scan outside of class
	# xs = jnp.zeros(N_steps)
	# (mom_fx, phi_fx), H_hist = lax.scan(leap_step, (mom_x0, phi_x0), xs = xs)
	# concat initil H0
	if record_H:
		H_hist = jnp.concatenate((H0[None], H_hist), axis=0)
		return mom_fx, phi_fx, H_hist
	return mom_fx, phi_fx
