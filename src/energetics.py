"""
Energetics functions for lattice field theory.
Intended to be used with HMC/lattice modules,
which handle error checking, layout inference, etc.
"""

import jax
import jax.numpy as jnp
from functools import partial
import params as params
from typing import Callable
jax.config.update("jax_enable_x64", True)


def phi4_action_core(phi_x: jnp.ndarray,
                     model: params.Phi4Params,
                     geom: params.LatticeGeometry,
                     shift: int,
                     spatial_axes: tuple[int, ...]
                     ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure numeric kernel: compute phi^4 action and pieces.

    Parameters
    ----------
    phi_x : jnp.ndarray
        Shape (V,) or (N, V) in some layout; `spatial_axes` selects the
        spatial dimensions (excluding any batch dim).
    model : Phi4Params
        Holds lam and kappa.
    geom : LatticeGeometry
        Holds D, etc.
    shift : int
        Index offset that tells where spatial axes start.
    spatial_axes : tuple[int, ...]
        Axes to sum over for spatial integrals.

    Returns
    -------
    S : jnp.ndarray
        Total action per configuration (shape () or (N,)).
    K : jnp.ndarray
        “Kinetic” / hopping part of the action, same shape as S.
    W : jnp.ndarray
        Neighbor interaction sum (essentially -K/kappa), same shape as S.
    """
    lam = model.lam
    kappa = model.kappa
    D = geom.D
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


def make_phi4_energy_fns(model: params.Phi4Params,
                         geom: params.LatticeGeometry,
                         shift: int,
                         spatial_axes: tuple[int, ...]
                         ) -> tuple[
                             Callable[[jnp.ndarray], jnp.ndarray],
                             Callable[[jnp.ndarray], jnp.ndarray],
                             Callable[[jnp.ndarray], jnp.ndarray]]:
    """
    Build energy functions for phi^4 theory:
      S_Fn(phi): per-config action (shape () or (N,))
      grad_S_Fn(phi): array same shape as phi, gradient of total action
      H_kinetic_Fn(mom): kinetic term 1/2 ∑_x p_x²
    """

    def S_Fn(phi_x):
        S, _, _ = phi4_action_core(phi_x, model, geom, shift, spatial_axes)
        return S

    def total_action_Fn(phi_x):
        # scalar required by jax.grad
        return S_Fn(phi_x).sum()

    grad_S_Fn = jax.grad(total_action_Fn)  # grad_S(phi) has same shape as phi
    # use like grad_S(phi_x)

    def H_kinetic_Fn(mom_x):
        # 1/2∑_x p_x²
        return (0.5 * (mom_x**2).sum(axis=spatial_axes))

    return S_Fn, grad_S_Fn, H_kinetic_Fn


def hamiltonian(phi_x: jnp.ndarray, mom_x: jnp.ndarray,
                model: params.Phi4Params,
                geom: params.LatticeGeometry,
                shift: int,
                spatial_axes: tuple[int, ...]
                ) -> jnp.ndarray:
    """
    Compute total Hamiltonian H = K + S for phi^4 theory.
    """
    S_fn, _, H_kinetic_fn = make_phi4_energy_fns(model,
                                                 geom,
                                                 shift,
                                                 spatial_axes)
    S = S_fn(phi_x)
    K = H_kinetic_fn(mom_x)
    return K + S
