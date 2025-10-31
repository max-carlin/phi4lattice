import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(3,4,5))
def action_core(phi_x, lam, kappa,  D, shift, spatial_axes):
    '''
    Pure JIT’d kernel
    calculate and return action and kinetic energy

    phi_x:
        either single field, shape = (phi_0, ...,phi_{D-1})
        or array of fields, shape = (N, phi_0, ...,phi_{D-1})

    Returns:
        S: action
        K: kinetic energy
        W: interaction term
        w/ shape = (N,) if phi_x is array of fields or scalar if single field
    '''
    # Eq 1.1:  S += -2 κ φ_x ∑_μ φ_{x+μ}  +  φ_x^2  +  λ(φ_x^2-1)^2

    K = 0
    for mu in range(D):
        ax = mu + shift
        # +mu -> +1; -mu -> -1, no need for factor of 2 in action/kinetic term
        K += (phi_x*(jnp.roll(phi_x,1,axis=ax)+jnp.roll(phi_x,-1,axis=ax))).sum(axis = spatial_axes)

    K *= - kappa #total kinetic
    U = (phi_x**2 + lam * (phi_x**2 -1.0)**2).sum(axis = spatial_axes) #total potential

    W = -K/kappa
    S = K + U
    return S, K, W

@partial(jax.jit, static_argnums=(1,2,3,4,5))
def grad_action_core(phi_x, lam, kappa, D, shift, spatial_axes):
    # total_action returns a scalar for jax.grad
    def total_action(phi):
        S_vals, _, _ = action_core(phi,
                                                lam, kappa,
                                                D, shift, spatial_axes)
        return jnp.sum(S_vals)
    #should compute grad(S) for both singular or batched configs w/out axis error
    #that occured from other method
    return jax.grad(total_action)(phi_x)



def grad_action(self, phi_x, lam, kappa, D, shift, spatial_axes):
    if phi_x is None:
        phi_x =phi_x
    return grad_action_core(phi_x, lam, kappa, D, shift, spatial_axes)

def _magnetization_core(phi_x, D):
    '''
    Pure JIT’d kernel
    returns array of magnetizations for each field configuration in phi_x
    '''
    m_array = phi_x.sum(axis = tuple(range(1,D+1)))
    return m_array

def magnetization(phi_x, D):
    '''
    returns array of magnetizations for each field configuration in phi_x
    '''
    return _magnetization_core(phi_x, D)

def binder_cumulant(phi_x, D):
    """
    returns Binder cumulant, but only for batched fields
    """
    m = magnetization(phi_x, D)
    m4_ave = (m**4).mean()
    m2_ave = (m**2).mean()
    return m4_ave/(m2_ave**2)

