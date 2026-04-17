import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=1)
def _magnetization_core(phi_x, spatial_axes, volume=None):
    '''
    Pure JIT’d kernel
    Returns array of magnetizations for each field configuration in phi_x
    '''
    
    m_array = phi_x.sum(axis=spatial_axes)
    if volume is not None:
        m_array = m_array / volume
    return m_array


def magnetization(phi_x, spatial_axes, volume=None):
    '''
    Returns array of magnetizations for each field configuration in phi_x
    '''
    return _magnetization_core(phi_x, spatial_axes, volume=volume)


def binder_cumulant(phi_x, spatial_axes, volume=None):
    """
    Returns Binder cumulant, but only for batched fields
    """
    m = magnetization(phi_x, spatial_axes, volume=volume)
    m4_ave = (m ** 4).mean()
    m2_ave = (m ** 2).mean()
    # schaefer's def:
    # return m4_ave / (m2_ave ** 2) 

    # standard def:
    return 1.0 - m4_ave / (3 * m2_ave ** 2)


def susceptibility(phi_x, spatial_axes, volume=None):
    m = magnetization(phi_x, spatial_axes, volume=volume)
    m_ave = m.mean()
    m2_ave = (m ** 2).mean()
    if volume is not None:
        susceptibility = volume * (m2_ave - m_ave ** 2)
    else:
        susceptibility = m2_ave - m_ave ** 2
    return susceptibility


def field_mean(phi_x, spatial_axes, volume=None):
    mean = phi_x.mean(axis=spatial_axes)
    if volume is not None:
        mean = mean / volume
    return mean


def field_variance(phi_x, spatial_axes, volume=None):
    mean = field_mean(phi_x, spatial_axes, volume=volume)
    mean2 = (phi_x ** 2).mean(axis=spatial_axes)
    variance = mean2 - mean ** 2
    return variance

