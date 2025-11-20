import jax
import jax.numpy as jnp
from functools import partial
import params


def phi4_action_core(phi_x: jnp.ndarray,
                     model: params.Phi4Params,
                     geom: params.LatticeGeometry,
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
                         spatial_axes: tuple[int, ...]):
    """
    Build:
      S_vals(phi): per-config action (shape () or (N,))
      grad_S(phi): array same shape as phi, gradient of total action
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
                spatial_axes: tuple[int, ...]):
    S_fn, _, H_kinetic_fn = make_phi4_energy_fns(model, geom, shift, spatial_axes)
    S = S_fn(phi_x)
    K = H_kinetic_fn(mom_x)
    return K + S


# # @staticmethod
# # @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
# def grad_action_core(phi_x, lam, kappa, D, shift, spatial_axes):
#     # total_action returns a scalar for jax.grad
#     def total_action(phi):
#         S_vals, _, _ = action_core(phi,
#                                    lam, kappa,
#                                    D, shift, spatial_axes)
#         return jnp.sum(S_vals)
#     # should compute grad(S) for both singular
#     # or batched configs w/out axis error
#     # that occured from other method
#     return jax.grad(total_action)(phi_x)


# def grad_action(self, phi_x, lam, kappa, D, shift, spatial_axes):
#     if phi_x is None:
#         phi_x = phi_x
#     return grad_action_core(phi_x, lam, kappa, D, shift, spatial_axes)


# @staticmethod
# @partial(jax.jit, static_argnums=1)
# def hamiltonian_kinetic_core(mom_x, spatial_axes):
#     # 1/2∑_x p_x²
#     return (0.5 * (mom_x**2).sum(axis=spatial_axes))


# def hamiltonian(phi_x: jnp.ndarray, mom_x: jnp.ndarray,
#                 lam: float, kappa: float,
#                 D: int, shift: int, spatial_axes: tuple[int]):
#     S, K, W = action_core(phi_x=phi_x,
#                           lam=lam,
#                           kappa=kappa,
#                           D=D,
#                           shift=shift,
#                           spatial_axes=spatial_axes)
#     mom_term = hamiltonian_kinetic_core(mom_x, spatial_axes)
#     return mom_term + S
