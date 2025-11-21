from typing import Dict
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import numbers
jax.config.update("jax_enable_x64", True)

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
    # seed for random momentum/metropolis criteria generation
    seed: int = 2  # default seed for hmc trajectories
    N_trajectories: int = 1  # number of HMC trajectories to run
    metropolis: bool = True  # whether to use Metropolis accept/reject
    record_H: bool = False  # whether to record Hamiltonian at each step
    verbose: bool = False  # whether to print progress information

    def __post_init__(self):
        '''
        validate integrator choice
        '''
        valid_integrators = ['leapfrog', 'omelyan']
        if self.integrator not in valid_integrators:
            raise ValueError(f"Invalid integrator: {self.integrator}. "
                             f"Choose from {valid_integrators}.")
        if not isinstance(self.N_steps, numbers.Integral) \
           or self.N_steps <= 0:
            raise ValueError("N_steps must be positive int.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if not isinstance(self.N_trajectories, numbers.Integral) \
           or self.N_trajectories <= 0:
            raise ValueError("N_trajectories must be positive int.")
        if not isinstance(self.metropolis, bool):
            raise TypeError("metropolis must be a boolean.")
        if not isinstance(self.record_H, bool):
            raise TypeError("record_H must be a boolean.")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean.")


@dataclass(frozen=True)
class LatticeGeometry:
    '''
    Lattice geometry parameters
    '''
    # array of spacing between lattice nodes in each D
    spacing_arr: jnp.ndarray
    # array of lattice lengths in each D
    length_arr: jnp.ndarray
    # number of spatial dimensions
    D: int = field(init=False)
    # total lattice volume
    V: int = field(init=False)
    # shape of the lattice in each D
    lat_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        '''
        initialization of derived geometric quantities
        '''
        # geom
        D = len(self.length_arr)
        V = jnp.prod(self.length_arr)
        # validate inputs
        if D <= 0:
            raise ValueError("Number of dimensions D must be positive.")
        if any(length <= 0 for length in self.length_arr):
            raise ValueError("All lattice lengths must be positive.")
        if any(spacing <= 0 for spacing in self.spacing_arr):
            raise ValueError("All lattice spacings must be positive.")
        if self.length_arr.shape != self.spacing_arr.shape:
            raise ValueError("length_arr and spacing_arr "
                             "must have the same shape.")
        # --- Sec1

        lat_shape = tuple((self.length_arr//self.spacing_arr).tolist())
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

    def __post_init__(self):
        '''
        validate physical parameters
        '''
        if not isinstance(self.lam, numbers.Real):
            raise TypeError("lam must be a float or int.")
        if not isinstance(self.kappa, numbers.Real):
            raise TypeError("kappa must be a float or int.")

# can add another params class for a differnt model
#  such as yukawa or gauge theory
