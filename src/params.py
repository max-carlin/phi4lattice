from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class HMCConfig:
    '''
    HMC simulation parameters
    '''
    eps: float  # integrator step size
    xi: float  # omelyan integrator parameter
    integrator: str  # 'leapfrog' or 'omelyan'
    N_steps: int  # number of integrator steps per trajectory
    metropolis: bool = True  # whether to use Metropolis accept/reject
    record_H: bool = False  # whether to record Hamiltonian at each step


@dataclass(frozen=True)
class Phi4Params:
    '''
    Physical parameters for phi^4 theory
    '''
    lam: float  # coupling constant (quartic interaction strength)
    kappa: float  # hopping parameter (related to mass term)


@dataclass(frozen=True)
class LatticeGeometry:
    '''
    Lattice geometry parameters
    '''
    L_array: jnp.ndarray  # array of lattice lengths in each dimension
    a_array: jnp.ndarray  # array of lattice spacings in each dimension
    lat_shape: tuple  # shape of the lattice
    D: int  # number of spatial dimensions
    V: int  # total number of lattice sites (volume)
    shift: int  # shift for batched computations
    spatial_axes: tuple  # axes corresponding to spatial dimensions

# geometry handled by lattice class
# can add another params class for a differnt model
#  such as yukawa or gauge theory


# # I actually don't see a reason to have this file.
# # hmc specific paramters are just:
# #    lam, kappa, xi, eps, N_steps, integrator, metropolis, & record_H
# # the rest are lattice geometry, physical params, or simulation params
# # this also mistypes some things like shift
# # and it removed the post init logic
# # we also don't necessarily want the hmc params
# # frozen in a separate class
# # we want them to be chosen at runtime
# # freezing would require creating a
# # new instance for each different set of params
# # which seems unnecessary
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
