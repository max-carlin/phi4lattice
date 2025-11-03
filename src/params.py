from dataclasses import dataclass


@dataclass(frozen=True)
class HMCParams:
    '''
    Dataclass that takes parameters from the lattice geometry.
    '''
    lam: float
    kappa: float
    D: int
    shift: tuple
    spatial_axes: tuple
    eps: float
    N_steps: int
    xi: float
    integrator: str
    metropolis: bool
    record_H: bool
    lat_shape: tuple
