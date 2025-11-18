import jax.numpy as jnp
import jax
from jax import jit  # use to translate to machine code for speed
from jax import random
from jax import lax
from functools import partial
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Dict
from .prng import init_fields
from .prng import make_keys
from .hmc import MD_traj
# from .params import HMCParams
import src.prng as prng
jax.config.update("jax_enable_x64", True)  # 64 bit


@dataclass(frozen=True)
class Phi4Lattice:
    '''Core class to handle and create our lattice.
t
    - Handles geometry, initialization

    Dataclass that takes parameters from the lattice geometry.


    Note on why dataclass and why frozen=True:
        JIT compilation is used heavily in this project for performance.

        data class auto-generates init, repr, eq, etc. methods.
        frozen=True makes it immutable and hashable:: __hash__ method is generated.
        Meaning once created, its attributes cannot be changed.
        This lets JAX use instances of this class as part of JIT-compiled functions
        because JAX uses static inputs to simplify JIT-compiled functions
        when caching and optimizing computations.

        "static" here means the value does not change between calls

        immutable -> hashable -> safe for static arguments in JAX JIT compilation.

        frozen should give value equality semantics,
        meaning two instances with the same attribute
        values are considered equal.

        multiple calls to functions with the same HMCParams instance
        should only compile once and reuse the compiled version when
        using JAX's JIT compilation.
    '''
    # ===============================================
    # defining param defaults and types
    # of user-settable parameters
    # using field() for those set in post_init

    # params w/ default vals can't go first in dataclass?
    # -----------------------------------------------

    # Physical Parameters ---------------------------
    kappa: float
    lam: float

    # Field Geometry --------------------------------
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
    
    # Field Configurations --------------------------
    phi_x: jnp.ndarray = field(init=False)
    mom_x: jnp.ndarray = field(init=False)
    spatial_axes: tuple = field(init=False)
    shift: int = field(init=False)
    # ===============================================

    # default prng (distribution) parameters --------
    mu: float = 0.0
    sigma: float = 0.1
    seed: int = 0
    n_keys: int = 1
    mom_seed: int = 1





    # ===============================================
    # Post init to compute derived quantities
    def __post_init__(self):
        '''
        initialization of geometric and field quantities
        '''
        # --- Sec1
        # geom
        D = len(self.L_array)
        V = jnp.prod(self.L_array)
        lat_shape = tuple((self.L_array//self.a_array).tolist())
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "V", V)
        object.__setattr__(self, "lat_shape", lat_shape)

        # single sample field configuration by default (seed = 0, n_keys =1)
        master_key = random.PRNGKey(self.seed)
        object.__setattr__(self, 'master_key', master_key)
        keys = random.split(master_key, self.n_keys)
        object.__setattr__(self, 'keys', keys)

        # partial pauses computation so vmap can use vectorized computation
        # by passing an array of keys simultaneously
        rng = partial(random.normal, shape=self.lat_shape, dtype=jnp.float64)
        object.__setattr__(self, 'rng', rng)
        phi_x = self.mu + self.sigma * jax.vmap(rng)(keys)
        object.__setattr__(self, "phi_x", phi_x)

        # ---Sec 2
        mom_master_key = random.PRNGKey(self.mom_seed)
        object.__setattr__(self, 'mom_master_key', mom_master_key)
        mom_keys = random.split(mom_master_key, self.n_keys)
        object.__setattr__(self, 'mom_keys', mom_keys)
        mom_x = jax.vmap(rng)(mom_keys)
        object.__setattr__(self, "mom_x", mom_x)

        spatial_axes = tuple(range(self.phi_x.ndim - self.D, self.phi_x.ndim))
        object.__setattr__(self, 'spatial_axes',
                           tuple(int(x) for x in spatial_axes))
        shift = self.phi_x.ndim - self.D
        object.__setattr__(self, 'shift', shift)

        object.__setattr__(self, 'H_history', None)

    # --------- Field Initialization Methods ---------
    @staticmethod
    def _rand_phi_core(keys,
                       lat_shape,
                       mu,
                       sigma,
                       dist):
        '''
        wrapper for prng randomization cores
        '''
        if dist == 'normal':
            return prng.randomize_normal_core(keys,
                                               lat_shape,
                                               mu,
                                               sigma)
        if dist == 'uniform':
            return prng.randomize_uniform_core(keys, lat_shape)


    def randomize_phi(self,
                      N,
                      s=0,
                      randomize_keys=True,
                      dist='normal') -> jnp.ndarray:
        """
        Host‐side
        generate N new keys, call the JIT’d kernel, then
        mutate self.phi_x on the Python side.
        """
        master_key, keys = make_keys(N, s, randomize_keys)

        # update master_key and keys
        object.__setattr__(self, "master_key", master_key)
        object.__setattr__(self, "keys", keys)

        # selecting dist type
        rand_phi_xs = self._rand_phi_core(keys,
                                          self.lat_shape,
                                          self.mu,
                                          self.sigma,
                                          dist)
        # elif dist == 'uniform':
        #     rand_phi_xs = self._randomize_uniform_core(keys, self.lat_shape)

        object.__setattr__(self, 'phi_x', rand_phi_xs)

        # determine if phi_x is singular or batched
        # if phi_x.ndim == self.D then spatial axes are (0,...,D-1)
        # if phi_x.ndim == self.D+1 then spatial axes are (1,...,D)
        #     N, the number of field configs, becomes the dimension 0
        # spatial_axes will give tup(rang(0,3))
        #     = (0,1,2) for single 3D field
        #  or = (1,2,3) for batched 3D fields
        spatial_axes = tuple(range(self.phi_x.ndim - self.D, self.phi_x.ndim))

        # shift will = 0 for single field; 1 for batch
        shift = self.phi_x.ndim - self.D

        # update spatial_axes and shift
        object.__setattr__(self, 'spatial_axes',
                           tuple(int(x) for x in spatial_axes))
        object.__setattr__(self, 'shift', shift)
        return self

    def randomize_mom(self, N, s=1, randomize_keys=True):
        mom_master_key, mom_keys = make_keys(N, s, randomize_keys)
        object.__setattr__(self, "mom_master_key", mom_master_key)
        object.__setattr__(self, "mom_keys", mom_keys)
        mom_xs = self._randomize_core(mom_keys, self.lat_shape, mu=0, sigma=1)
        object.__setattr__(self, 'mom_x', mom_xs)
        return self

    # use for constant fields for simple magnetization expectations
    def constant_phi(self, constant):
        '''
        set all phi_x values to constant
        '''
        phi_x = jnp.full_like(self.phi_x, constant, dtype=np.float64)
        object.__setattr__(self, 'phi_x', phi_x)
        return self

    def constant_momentum(self, constant):
        '''
        set all phi_x values to constant
        '''
        mom_x = jnp.full_like(self.mom_x, constant, dtype=np.float64)
        object.__setattr__(self, 'mom_x', mom_x)
        return self
    # --------------------------------------------------

    def _split_keys(self, n):
        keys = random.split(self.master_key, n + 1)
        master_key, subkeys = keys[0], keys[1:]
        object.__setattr__(self, "master_key", master_key)
        return subkeys

    def HMC(self,
            N_steps,
            eps,
            xi,
            integrator='omelyan',
            s=0,
            N_trajectories=1,
            metropolis=True,
            record_H=False,
            verbose=False,
            *, measure_fns=None):

        params = HMCParams(
                lam=self.lam,
                kappa=self.kappa,
                D=self.D,
                shift=self.shift,
                spatial_axes=self.spatial_axes,
                eps=eps,
                N_steps=N_steps,
                xi=xi,
                integrator=integrator,
                metropolis=metropolis,
                record_H=record_H,
                lat_shape=self.lat_shape)

        # master_key = random.PRNGKey(np.random.randint(0,10**6))

        # need split_keys to get subkeys from the same master key
        traj_keys = self._split_keys(2 * N_trajectories)
        traj_keys = traj_keys.reshape((N_trajectories, 2, 2))

        # I don't know if this kind of nested function is okay,
        # but this is the only solution I can think of.
        # measure_fns can't be in params.py because because
        # a dict is not hashable to JAX
        def one_traj(state, key_pair):
            return MD_traj(state, key_pair, params, measure_fns)

        # need lambda key because jax.scan only takes
        # (carry, element) NOT params
        (mom_accepted, phi_accepted), out_dict = lax.scan(
            one_traj,
            (self.mom_x, self.phi_x),
            xs=traj_keys,
            length=N_trajectories
        )

        object.__setattr__(self, 'mom_x', mom_accepted)
        object.__setattr__(self, 'phi_x', phi_accepted)

        if record_H or measure_fns or verbose:
            object.__setattr__(self, 'measure_history', out_dict)

        return self

    def _magnetization_core(phi_x, D):
        '''
        Pure JIT’d kernel
        returns array of magnetizations for each field configuration in phi_x
        '''
        m_array = phi_x.sum(axis=tuple(range(1, D + 1)))
        return m_array
