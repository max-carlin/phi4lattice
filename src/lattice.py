import jax.numpy as jnp
import jax
from jax import random
from jax import lax
from functools import partial
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Dict, Optional
from project.phi4lattice_folder.src.layout_utils import infer_layout
from .prng import make_keys
import src.hmc as hmc
import src.prng as prng
import src.params as params
import src.energetics as eng
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

        # infer layout
        spatial_axes, shift = infer_layout(self.phi_x, self.geom.D)
        object.__setattr__(self, 'spatial_axes',
                           tuple(int(x) for x in spatial_axes))
        object.__setattr__(self, 'shift', shift)

        # history initialization
        object.__setattr__(self, 'H_history', None)

    # ===============================================
    # --------- Field Initialization Methods --------
    # ===============================================
    # ----- phi field methods ------
    @staticmethod
    def _rand_field_core(keys,
                         lat_shape,
                         mu = None,
                         sigma = None,
                         dist = 'normal') -> jnp.ndarray:
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

        # infer layout
        spatial_axes, shift = infer_layout(self.phi_x, self.geom.D)
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
    def randomize_mom(self,
                      N,
                      s=None,
                      randomize_keys=True,
                      dist='normal'):
        mom_master_key, mom_keys = make_keys(N, s, randomize_keys)
        object.__setattr__(self, "mom_master_key", mom_master_key)
        object.__setattr__(self, "mom_keys", mom_keys)
        mom_xs = self._rand_field_core(mom_keys, self.geom.lat_shape,
                                       mu=0, sigma=1, dist='normal')
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
    def _next_traj_keys(self, n_traj: int, *, seed: int | None = None) -> jnp.ndarray:
        """
        Return array of shape (n_traj, 2, 2) with (key_mom, key_meta) per trajectory.
        If seed is given, use that; otherwise advance self.hmc_master_key.
        """
        if seed is not None:
            key = random.PRNGKey(int(seed))
        else:
            key, new_key = random.split(self.hmc_config.seed)
            object.__setattr__(self, "hmc_master_key", new_key)

        traj_keys = random.split(key, 2 * n_traj).reshape(n_traj, 2, 2)
        return traj_keys

    def run_HMC(self,
                cfg: params.HMCConfig,
                seed: int = None,
                measure_fns: Dict[str, callable] = None
                ):
        phi0=self.phi_x
        mom0=self.mom_x
        if seed is None:
            seed = self.hmc_config.seed
        traj_keys = self._next_traj_keys(cfg.N_trajectories, seed=seed)

        energy_fns = eng.make_phi4_energy_fns
        S_Fn, grad_S_Fn, H_kinetic_Fn = energy_fns(self.model,
                                                   self.geom,
                                                   self.shift,
                                                   self.spatial_axes)

        hmc.run_HMC_trajectories(phi0=phi0,
                                 mom0=mom0,
                                 traj_keys=traj_keys,
                                 cfg=cfg,
                                 S_Fn=S_Fn,
                                 grad_S_Fn=grad_S_Fn,
                                 H_kinetic_Fn=H_kinetic_Fn,
                                 measure_fns=measure_fns)
