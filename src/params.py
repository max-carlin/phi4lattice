from typing import Dict
from dataclasses import dataclass, field
import jax.numpy as jnp


"""
Note on why dataclass and why frozen=True:
    JIT compilation is used heavily in this project for performance.

    data class auto-generates init, repr, eq, etc. methods.
    frozen=True makes it immutable and hashable:
        __hash__ method is generated.
    Meaning once created, its attributes cannot be changed.
    This lets JAX use JIT-compiled functions
        JAX uses static inputs to simplify JIT-compiled functions
        when caching and optimizing computations.

    "static" here means the value does not change between calls

    immutable -> hashable -> safe for static arguments in JIT compilation.

    frozen should give value equality semantics,
    meaning two instances with the same attribute
    values are considered equal.

    multiple calls to functions with the same HMCParams instance
    should only compile once and reuse the compiled version when
    using JAX's JIT compilation.
"""


@dataclass(frozen=True)
class HMCConfig:
    '''
    HMC simulation parameters
    '''
    N_steps: int  # number of integrator steps per trajectory
    eps: float  # integrator step size
    xi: float  # omelyan integrator parameter
    integrator: str  # 'leapfrog' or 'omelyan'
    # seed for random momentum generation
    seed: int = 2,  # default seed for hmc trajectories
    N_trajectories: int = 1  # number of HMC trajectories to run
    metropolis: bool = True  # whether to use Metropolis accept/reject
    record_H: bool = False  # whether to record Hamiltonian at each step
    verbose: bool = False


@dataclass(frozen=True)
class LatticeGeometry:
    '''
    Lattice geometry parameters
    '''
    # array of spacing between lattice nodes in each D
    a_array: jnp.ndarray
    # array of lattice lengths in each D
    L_array: jnp.ndarray
    # number of spatial dimensions
    D: int = field(init=False)
    # total lattice volume
    V: int = field(init=False)
    # shape of the lattice in each D
    lat_shape: jnp.ndarray = field(init=False)

    def __post_init__(self):
        '''
        initialization of derrived geometric quantities
        '''
        # --- Sec1
        # geom
        D = len(self.L_array)
        V = jnp.prod(self.L_array)
        lat_shape = tuple((self.L_array//self.a_array).tolist())
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "V", V)
        object.__setattr__(self, "lat_shape", lat_shape)


@dataclass(frozen=True)
class Phi4Params:
    '''
    Physical parameters for phi^4 theory
    '''
    lam: float  # coupling constant (quartic interaction strength)
    kappa: float  # hopping parameter (related to mass term)

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
