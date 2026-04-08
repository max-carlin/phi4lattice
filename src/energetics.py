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

# ------------ Phi^4 model energetics -------


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
    # W is the neighbor interaction sum, essentially -K/kappa
    W = 0
    for mu in range(D):
        ax = mu + shift
        # +mu -> +1; -mu -> -1, no need for factor of 2 in action/kinetic term
        W += (phi_x * (jnp.roll(phi_x, 1, axis=ax)
              + jnp.roll(phi_x, -1, axis=ax))).sum(axis=spatial_axes)

    K = - kappa * W  # total kinetic
    # total potential
    U = (phi_x ** 2
         + lam * (phi_x ** 2 - 1.0) ** 2).sum(axis=spatial_axes)

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
      H_kinetic_Fn(mom): kinetic term 1/2 ∑_x p_x²;
      fictional kinetic energy for HMC momentum updates
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
        """Fictional kinetic energy for HMC momentum updates;
        not a physical kinetic energy."""
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
    K_fictious = H_kinetic_fn(mom_x)
    return K_fictious + S

# ------------ Ising model energetics -------


def ising_action_core(sigma_x: jnp.ndarray,
                      model: params.IsingParams,
                      geom: params.LatticeGeometry,
                      shift: int,
                      spatial_axes: tuple[int, ...]
                      ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure numeric kernel: compute Ising action.
    definition: S = -κ ∑_{x,μ} σ_x σ_{x+μ} - h ∑_x σ_x
    with S = beta * H, and H = -J ∑_{x,μ} σ_x σ_{x+μ} - h' ∑_x σ_x
    Parameters
    ----------
    sigma_x : jnp.ndarray
        Shape (V,) or (N, V) in some layout; `spatial_axes` selects the
        spatial dimensions (excluding any batch dim).
    model : IsingParams
        Holds kappa and h.
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
    """
    kappa = model.kappa
    h = model.h
    D = geom.D
    # S = -κ ∑_{x,μ} σ_x σ_{x+μ} - h ∑_x σ_x
    # W is the neighbor interaction sum, essentially -K/kappa
    W = 0
    for mu in range(D):
        ax = mu + shift
        W += (sigma_x * jnp.roll(sigma_x, 1, axis=ax)).sum(axis=spatial_axes)

    # total kinetic
    K = -kappa * W  # bond interaction part of action
    # total potential/external field part of action
    U = - h * sigma_x.sum(axis=spatial_axes)

    S = K + U
    return S, K, W


def make_ising_energy_fns(model: params.IsingParams,
                          geom: params.LatticeGeometry,
                          shift: int,
                          spatial_axes: tuple[int, ...]
                          ) -> tuple[
                              Callable[[jnp.ndarray], jnp.ndarray],
                              Callable[[jnp.ndarray], jnp.ndarray]]:
    """
    Build energy function for Ising theory:
      S_Fn(sigma): per-config action (shape () or (N,))
    """

    def S_Fn(sigma_x):
        S, _, _ = ising_action_core(sigma_x, model, geom, shift, spatial_axes)
        return S

    def propose_flip_Fn(sigma_x: jnp.ndarray,
                        site_key: jnp.ndarray,
                        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        # split site_key into coordinate keys for each dimension
        coord_keys = jax.random.split(site_key, geom.D)

        if shift == 0:
            # not batched
            site_coords = tuple(
                jax.random.randint(coord_keys[mu],
                                   shape=(),
                                   minval=0,
                                   maxval=geom.lat_shape[mu])
                for mu in range(geom.D)
                )
            old_spin = sigma_x[site_coords]
            # flip spin at site
            sigma_prop = sigma_x.at[site_coords].set(-old_spin)

        else:
            # batched; site_key is shape (N, D)
            batch_size = sigma_x.shape[0]
            site_coords = tuple(
                jax.random.randint(coord_keys[mu],
                                   shape=(batch_size,),
                                   minval=0,
                                   maxval=geom.lat_shape[mu])
                for mu in range(geom.D)
            )
            # gather old spins at site for each config in batch
            batch_idxs = jnp.arange(batch_size)
            full_coords = (batch_idxs, *site_coords)
            old_spins = sigma_x[full_coords]
            # flip spins at site for each config
            sigma_prop = sigma_x.at[full_coords].set(-old_spins)
        return sigma_prop, site_coords
    return S_Fn, propose_flip_Fn
