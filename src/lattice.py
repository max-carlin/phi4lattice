import jax.numpy as jnp
import jax
from jax import random
from jax import lax
from functools import partial
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Dict, Optional
from .prng import make_keys
import src.hmc as hmc
import src.prng as prng
import src.params as params
jax.config.update("jax_enable_x64", True)  # 64 bit


@dataclass
class Phi4Lattice:
    '''Core class to handle and create our lattice.
    - Handles geometry, initialization

    Dataclass that takes parameters from the lattice geometry.
    '''
    # ===============================================
    # ------- defining param defaults and types -----
    # ===============================================
    # of user-settable parameters
    # using field() for those set in post_init
    # -----------------------------------------------
    # Physical Parameters ---------------------------
    # --- new API uses model param class ---
    model: Optional[params.Phi4Params] = None  # frozen, hashable
    # Field Geometry --------------------------------
    geom: Optional[params.LatticeGeometry] = None  # frozen, hashable
    # hmc configuration ------------------------------
    hmc_config: Optional[params.HMCConfig] = None  # frozen, hashable
    # --- Legacy API (for tests / old code) ---
    a_array: Optional[jnp.ndarray] = None
    L_array: Optional[jnp.ndarray] = None
    kappa: Optional[float] = None
    lam: Optional[float] = None

    # mutable parameters/state
    # Field/State Configurations --------------------------
    phi_x: jnp.ndarray = field(init=False)
    mom_x: jnp.ndarray = field(init=False)
    spatial_axes: tuple = field(init=False)
    shift: int = field(init=False)

    H_history: jnp.ndarray = field(init=False)  # Hamiltonian history
    measure_history: Dict[str, jnp.ndarray] = field(init=False,
                                                    default_factory=dict)

    # default prng (distribution) parameters --------
    mu: float = 0.0
    sigma: float = 0.1
    phi_seed: int = 0
    n_keys: int = 1
    mom_seed: int = 1

    # ===============================================
    # ------ post init computations -------
    # ===============================================
    # Post init to compute derived quantities
    def __post_init__(self):
        '''
        initialization of lattice specific quantities
        and rng params
        '''
        # legacy API support
        if self.model is None:
            if self.kappa is None or self.lam is None:
                raise ValueError("Either provide model params "
                                 "or kappa and lam.")
            self.model = params.Phi4Params(lam=self.lam, kappa=self.kappa)
        if self.geom is None:
            if self.a_array is None or self.L_array is None:
                raise ValueError("Either provide geometry params "
                                 "or a_array and L_array.")
            self.geom = params.LatticeGeometry(a_array=self.a_array,
                                               L_array=self.L_array)

        # single sample field configuration by default (seed = 0, n_keys =1)
        phi_master_key = random.PRNGKey(self.phi_seed)
        object.__setattr__(self, 'phi_master_key', phi_master_key)
        phi_keys = random.split(phi_master_key, self.n_keys)
        object.__setattr__(self, 'phi_keys', phi_keys)
        # partial pauses computation so vmap can use vectorized computation
        # by passing an array of keys simultaneously
        rng = partial(random.normal,
                      shape=self.geom.lat_shape,
                      dtype=jnp.float64)

        # initialize phi field configuration
        object.__setattr__(self, 'rng', rng)
        phi_x = self.mu + self.sigma * jax.vmap(rng)(phi_keys)
        object.__setattr__(self, "phi_x", phi_x)

        # initialize momentum field
        mom_master_key = random.PRNGKey(self.mom_seed)
        object.__setattr__(self, 'mom_master_key', mom_master_key)
        mom_keys = random.split(mom_master_key, self.n_keys)
        object.__setattr__(self, 'mom_keys', mom_keys)
        mom_x = jax.vmap(rng)(mom_keys)
        object.__setattr__(self, "mom_x", mom_x)

        # determine if phi_x is singular or batched
        # if phi_x.ndim == self.D then spatial axes are (0,...,D-1)
        # if phi_x.ndim == self.D+1 then spatial axes are (1,...,D)
        #     N, the number of field configs, becomes the dimension 0
        # spatial_axes will give tup(rang(0,3))
        #     = (0,1,2) for single 3D field
        #  or = (1,2,3) for batched 3D fields
        spatial_axes = tuple(range(self.phi_x.ndim - self.geom.D,
                                   self.phi_x.ndim))
        object.__setattr__(self, 'spatial_axes',
                           tuple(int(x) for x in spatial_axes))
        shift = self.phi_x.ndim - self.geom.D
        object.__setattr__(self, 'shift', shift)

        # history initialization
        object.__setattr__(self, 'H_history', None)

    # ===============================================
    # --------- Field Initialization Methods --------
    # ===============================================
    # ----- phi field methods ------
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
        object.__setattr__(self, "phi_master_key", master_key)
        object.__setattr__(self, "phi_keys", keys)

        # selecting dist type
        rand_phi_xs = self._rand_phi_core(keys,
                                          self.geom.lat_shape,
                                          self.mu,
                                          self.sigma,
                                          dist)

        object.__setattr__(self, 'phi_x', rand_phi_xs)

        # determine if phi_x is singular or batched
        # if phi_x.ndim == self.D then spatial axes are (0,...,D-1)
        # if phi_x.ndim == self.D+1 then spatial axes are (1,...,D)
        #     N, the number of field configs, becomes the dimension 0
        # spatial_axes will give tup(rang(0,3))
        #     = (0,1,2) for single 3D field
        #  or = (1,2,3) for batched 3D fields
        spatial_axes = tuple(range(self.phi_x.ndim - self.geom.D,
                                   self.phi_x.ndim))

        # shift will = 0 for single field; 1 for batch
        shift = self.phi_x.ndim - self.geom.D

        # update spatial_axes and shift
        object.__setattr__(self, 'spatial_axes',
                           tuple(int(x) for x in spatial_axes))
        object.__setattr__(self, 'shift', shift)
        return self

    # use for constant fields for simple magnetization expectations
    def constant_phi(self, constant):
        '''
        set all phi_x values to constant
        '''
        phi_x = jnp.full_like(self.phi_x, constant, dtype=np.float64)
        object.__setattr__(self, 'phi_x', phi_x)
        return self

    # --- mom field methods -------
    def randomize_mom(self, N, s=1, randomize_keys=True):
        mom_master_key, mom_keys = make_keys(N, s, randomize_keys)
        object.__setattr__(self, "mom_master_key", mom_master_key)
        object.__setattr__(self, "mom_keys", mom_keys)
        mom_xs = self._randomize_core(mom_keys, self.geom.lat_shape,
                                      mu=0, sigma=1)
        object.__setattr__(self, 'mom_x', mom_xs)
        return self

    def constant_momentum(self, constant):
        '''
        set all phi_x values to constant
        '''
        mom_x = jnp.full_like(self.mom_x, constant, dtype=np.float64)
        object.__setattr__(self, 'mom_x', mom_x)
        return self

    # ===============================================
    # ------------- HMC field evolution -------------
    # ===============================================
    def _split_keys(self, n):
        keys = random.split(self.master_key, n + 1)
        master_key, subkeys = keys[0], keys[1:]
        object.__setattr__(self, "master_key", master_key)
        return subkeys

    def HMC(self,
            cfg_or_N_steps: params.HMCConfig | int,
            eps: float = None,
            xi: float = None,
            integrator: str = None,
            seed: int = None,  # if seed is None, random seed is used
            N_trajectories: int = 1,
            metropolis: bool = True,
            record_H: bool = False,
            verbose: bool = False,
            *, measure_fns: dict = None):
        """
        Run N HMC tractories

        Two call styles supported
        New API:
        1) HMC(cfg: HMCConfig, N_trajectories: int, ...)
           where cfg is a HMCConfig dataclass instance

        Legacy API:
        2) HMC(N_steps=..., eps=..., xi=...,
              integrator=..., N_trajectories=..., ...)
              where N_steps, eps, xi, etc are passed directly
        """
        # parse HMC config
        if isinstance(cfg_or_N_steps, params.HMCConfig):
            if eps is not None or xi is not None:
                raise ValueError("When using HMCConfig, "
                                 "eps and xi should not be provided.")
            if integrator != 'leapfrog':
                raise ValueError("When using HMCConfig, "
                                 "integrator should not be provided.")
            cfg = cfg_or_N_steps
            N_steps = cfg.N_steps
            eps = cfg.eps
            xi = cfg.xi
            integrator = cfg.integrator
            seed = cfg.seed
            N_trajectories = cfg.N_trajectories
            metropolis = cfg.metropolis
            record_H = cfg.record_H
            verbose = cfg.verbose
        elif isinstance(cfg_or_N_steps, int):
            N_steps = cfg_or_N_steps
            if eps is None:
                raise ValueError("When using legacy HMC API, "
                                 "eps must be provided.")
            if integrator is None:
                raise ValueError("When using legacy HMC API, "
                                 "integrator must be provided.")
            if integrator == 'omelyan' and xi is None:
                raise ValueError("When using legacy HMC API with "
                                 "omelyan integrator, xi must be provided.")
            cfg = params.HMCConfig(N_steps=N_steps,
                                   eps=eps,
                                   xi=xi,
                                   integrator=integrator,
                                   seed=seed,
                                   N_trajectories=N_trajectories,
                                   metropolis=metropolis,
                                   record_H=record_H,
                                   verbose=verbose)
        else:
            raise ValueError("First argument to HMC must be "
                             "either HMCConfig or int (N_steps).")

        # create mom rng keys
        if cfg.seed is not None:
            if isinstance(cfg.seed, int):
                md_master_key = random.PRNGKey(cfg.seed)
            # if seed is already a prng_key
            if isinstance(cfg.seed, jnp.ndarray):
                md_master_key = cfg.seed
        else:
            md_master_key = random.PRNGKey(np.random.randint(0, 10**6))

        # need split_keys to get subkeys from the same master key
        traj_keys = self._split_keys(md_master_key, 2 * cfg.N_trajectories)
        # reshape for [(mom_key_1, r_key_1),...,(mom_key_N, r_key_N)]
        traj_keys = traj_keys.reshape((cfg.N_trajectories, 2, 2))

        # one trajectory function for lax.scan
        def one_traj(state, key_pair):
            # one key pair = (mom_key, r_key)
            # one for mom refresh and one for metropolis test
            return hmc.MD_traj(state, key_pair, cfg, measure_fns)

        # need lambda key because lax.scan only takes
        # (carry, element) NOT params
        (mom_accepted, phi_accepted), out_dict = lax.scan(
            one_traj,
            (self.mom_x, self.phi_x),
            xs=traj_keys,
            length=cfg.N_trajectories
        )

        object.__setattr__(self, 'mom_x', mom_accepted)
        object.__setattr__(self, 'phi_x', phi_accepted)

        if record_H or measure_fns or verbose:
            object.__setattr__(self, 'measure_history', out_dict)

        return self
