import jax
import jax.numpy as jnp
from functools import partial
from .params import LatticeGeometry
from .params import Phi4Params


@staticmethod
@partial(jax.jit, static_argnums=(3, 4, 5))
def phi4_action_core(phi_x: jnp.ndarray,
                     lam: float, kappa: float,
                     D: int,
                     shift: int,
                     spatial_axes: tuple):
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
        K += (phi_x * (jnp.roll(phi_x, 1, axis=ax)
              + jnp.roll(phi_x, -1, axis=ax))).sum(axis=spatial_axes)

    K *= - kappa  # total kinetic
    # total potential
    U = (phi_x ** 2
         + lam * (phi_x ** 2 - 1.0) ** 2).sum(axis=spatial_axes)

    W = -K/kappa
    S = K + U
    return S, K, W


@staticmethod
@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def grad_action_core(phi_x, lam, kappa, D, shift, spatial_axes):
    # total_action returns a scalar for jax.grad
    def total_action(phi):
        S_vals, _, _ = action_core(phi,
                                   lam, kappa,
                                   D, shift, spatial_axes)
        return jnp.sum(S_vals)
    # should compute grad(S) for both singular
    # or batched configs w/out axis error
    # that occured from other method
    return jax.grad(total_action)(phi_x)


def grad_action(self, phi_x, lam, kappa, D, shift, spatial_axes):
    if phi_x is None:
        phi_x = phi_x
    return grad_action_core(phi_x, lam, kappa, D, shift, spatial_axes)


@staticmethod
@partial(jax.jit, static_argnums=1)
def hamiltonian_kinetic_core(mom_x, spatial_axes):
    # 1/2∑_x p_x²
    return (0.5 * (mom_x**2).sum(axis=spatial_axes))


def hamiltonian(phi_x: jnp.ndarray, mom_x: jnp.ndarray,
                lam: float, kappa: float,
                D: int, shift: int, spatial_axes: tuple[int]):
    S, K, W = action_core(phi_x=phi_x,
                          lam=lam,
                          kappa=kappa,
                          D=D,
                          shift=shift,
                          spatial_axes=spatial_axes)
    mom_term = hamiltonian_kinetic_core(mom_x, spatial_axes)
    return mom_term + S
