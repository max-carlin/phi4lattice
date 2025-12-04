import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=1)
def _magnetization_core(phi_x, D):
    '''
    Pure JIT’d kernel
    Returns array of magnetizations for each field configuration in phi_x
    '''
    m_array = phi_x.sum(axis=tuple(range(1, D+1)))
    return m_array


def magnetization(phi_x, D):
    '''
    Returns array of magnetizations for each field configuration in phi_x
    '''
    return _magnetization_core(phi_x, D)


def binder_cumulant(phi_x, D):
    """
    Returns Binder cumulant, but only for batched fields
    """
    m = magnetization(phi_x, D)
    m4_ave = (m ** 4).mean()
    m2_ave = (m ** 2).mean()
    return m4_ave / (m2_ave ** 2)
