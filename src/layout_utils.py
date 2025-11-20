import jax
import jax.numpy as jnp

def infer_layout(phi_x: jnp.ndarray, D: int):
    '''
    Given a field configuration array phi_x and the spatial dimension D,
    determines whether phi_x represents a single field configuration
    or a batch of configurations. Returns the shift (0 for single field,
    1 for batch) and the spatial axes tuple.
    '''
    # determine if phi_x is singular or batched
    # if phi_x.ndim == self.D then spatial axes are (0,...,D-1)
    # if phi_x.ndim == self.D+1 then spatial axes are (1,...,D)
    #     N, the number of field configs, becomes the dimension 0
    # spatial_axes will give tup(rang(0,3))
    #     = (0,1,2) for single 3D field
    #  or = (1,2,3) for batched 3D fields
    spatial_axes = tuple(range(phi_x.ndim - D, phi_x.ndim))
    # shift will = 0 for single field; 1 for batch
    shift = phi_x.ndim - D
    return shift, spatial_axes