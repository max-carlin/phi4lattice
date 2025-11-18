from dataclasses import dataclass


# I actually don't see a reason to have this file.
# hmc specific paramters are just:
#    lam, kappa, xi, eps, N_steps, integrator, metropolis, & record_H
# the rest are lattice geometry, physical params, or simulation params
# this also mistypes some things like shift
# and it removed the post init logic
# we also don't want the hmc params frozen in a separate class
# we want them to be chosen at runtime
@dataclass(frozen=True)
class HMCParams:
    '''
    '''
    # lambda is the coupling constant
    #   also known as the quartic (self) interaction strength
    lam: float
    # kappa is the hopping parameter
    #   relates to the mass term in the action
    kappa: float
    D: int  # number of spatial dimensions

    # if batched, shift
    shift: tuple
    spatial_axes: tuple
    eps: float
    N_steps: int
    xi: float
    integrator: str
    metropolis: bool
    record_H: bool
    lat_shape: tuple
