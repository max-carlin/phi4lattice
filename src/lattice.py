import jax.numpy as jnp
import jax
from jax import random
from jax import lax
from functools import partial
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Dict, Optional
from layout_utils import infer_layout
from prng import make_keys
import hmc as hmc
import prng as prng
import params as params
import energetics as eng
import numbers
jax.config.update("jax_enable_x64", True)  # 64 bit


@dataclass
class Phi4Lattice:
    '''Core class to handle and create the lattice.

    This class has methods to handle:
    - Lattice geometry.
    - Model parameters for phi-4 theory.
    - Initialization of phi and momentum fields.
    - HMC evolution of the field.
    - PRNG key management.

    Parameters
    ----------
    model : params.Phi4Params or None
        Frozen dataclass containing coupling constant and
        hopping parameter. If not provided, the
        parameters `kappa` and `lam` must be supplied.
    geom : params.LatticeGeometry or None
        Frozen dataclass describing lattice spacing and shape. If not provided,
        the parameters `spacing_arr` and `length_arr` must be supplied.
    hmc_config : params.HMCConfig or None
        Optional default HMC configuration.
    spacing_arr, length_arr : array-like, optional
        geometry parameters.
    kappa, lam : float, optional
        phi-4 model parameters if not initializing Phi4Params.
    phi_dist, phi_mu, phi_sigma, phi_seed : various
        Distribution parameters for phi initialization.
    mom_dist, mom_mu, mom_sigma, mom_seed : various
        Distribution parameters for momentum initialization.
    n_keys : int
        Number of independent field configurations (PRNG batch size).
    valid_dists : tuple
        Allowed string values for PRNG distributions ("normal", "uniform").

    Attributes
    ----------
    phi_x : jnp.ndarray
        Current field.
    mom_x : jnp.ndarray
        Current momentum field.
    spatial_axes : tuple[int]
        Axes as spatial dimensions.
    shift : int
        Lattice shift parameter.
    trajectory_history : dict[str, jnp.ndarray]
        Dictionary storing HMC trajectory measurements.
    hmc_master_key : jnp.ndarray
        Master PRNG key for HMC sampling.
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
    spacing_arr: Optional[jnp.ndarray] = None
    length_arr: Optional[jnp.ndarray] = None
    kappa: Optional[float] = None
    lam: Optional[float] = None

    # mutable parameters/state
    # Field/State Configurations --------------------------
    phi_x: jnp.ndarray = field(init=False)
    mom_x: jnp.ndarray = field(init=False)
    spatial_axes: tuple = field(init=False)
    shift: int = field(init=False)

    # H_history: jnp.ndarray = field(init=False)  # Hamiltonian history
    # measure_history: Dict[str, jnp.ndarray] = field(init=False,
    #                                                 default_factory=dict)
    trajectory_history: Dict[str, jnp.ndarray] = field(init=False,
                                                       default_factory=dict)
    # default prng (distribution) parameters --------
    valid_dists: tuple = field(init=False,
                               default=('normal', 'uniform'))
    # phi field defaults to normal distribution
    phi_dist: str = 'normal'
    phi_mu: float | None = None
    phi_sigma: float | None = None
    phi_seed: int | jnp.ndarray | None = 0
    n_keys: int = 1  # batch size (independent field configs)

    # momentum field defaults to normal distribution
    mom_dist: str = 'normal'
    mom_mu: float | None = None
    mom_sigma: float | None = None
    mom_seed: int | jnp.ndarray | None = 1

    hmc_master_key: jnp.ndarray = field(init=False)

    # ===============================================
    # ------ post init computations -------
    # ===============================================
    # Post init to compute derived quantities
    def __post_init__(self):
        '''
        Post-initialization tasks for lattice specific
        quantities and RNG parameters.
        '''
        # legacy API support
        if self.model is None:
            if self.kappa is None or self.lam is None:
                raise ValueError("Either provide model params "
                                 "or kappa and lam.")
            self.model = params.Phi4Params(lam=self.lam, kappa=self.kappa)
        if self.geom is None:
            if self.spacing_arr is None or self.length_arr is None:
                raise ValueError("Either provide geometry params "
                                 "or spacing_arr and length_arr.")
            self.geom = params.LatticeGeometry(spacing_arr=self.spacing_arr,
                                               length_arr=self.length_arr)

        # initialize phi field configuration
        # single sample field configuration by default (seed = 0, n_keys =1)
        if not isinstance(self.n_keys, numbers.Integral) or self.n_keys <= 0:
            raise ValueError("n_keys must be positive integer.")

        phi_master_key, phi_keys = make_keys(
                                        self.n_keys, self.phi_seed,
                                        randomize_keys=(self.phi_seed is None))
        phi_x = self._rand_field_core(phi_keys,
                                      self.geom.lat_shape,
                                      mu=self.phi_mu,
                                      sigma=self.phi_sigma,
                                      dist=self.phi_dist)
        object.__setattr__(self, 'phi_master_key', phi_master_key)
        object.__setattr__(self, 'phi_keys', phi_keys)
        object.__setattr__(self, "phi_x", phi_x)

        # initialize momentum field
        mom_master_key, mom_keys = make_keys(
                                        self.n_keys, self.mom_seed,
                                        randomize_keys=(self.mom_seed is None))
        mom_x = self._rand_field_core(mom_keys,
                                      self.geom.lat_shape,
                                      mu=self.mom_mu,
                                      sigma=self.mom_sigma,
                                      dist=self.mom_dist)
        object.__setattr__(self, 'mom_master_key', mom_master_key)
        object.__setattr__(self, 'mom_keys', mom_keys)
        object.__setattr__(self, "mom_x", mom_x)

        # infer layout
        spatial_axes, shift = infer_layout(self.phi_x, self.geom.D)
        object.__setattr__(self, 'spatial_axes',
                           tuple(int(x) for x in spatial_axes))
        object.__setattr__(self, 'shift', shift)

    # ===========================================================
    # --------- Field Initialization Methods --------
    # other than _rand_field_core, these return self for chaining
    # ===========================================================
    # ----- phi field methods ------------
    @staticmethod
    def _rand_field_core(keys,
                         lat_shape,
                         mu=None,
                         sigma=None,
                         dist='normal') -> jnp.ndarray:
        ''' Wrapper for PRNG randomization cores.

        Parameters
        ----------
        keys : jnp.ndarray
            Array of PRNG keys (one per independent configuration).
        lat_shape : tuple[int]
            Lattice shape.
        mu, sigma : float or None
            Mean and standard deviation for normal distributions.
        dist : {'normal', 'uniform'}
            Choice of PRNG distribution.

        Returns
        -------
        jnp.ndarray
            Random field array with shape (len(keys), *lat_shape).
        '''
        # valid_dists = ['normal', 'uniform']
        if dist == 'normal':
            return prng.randomize_normal_core(keys,
                                              lat_shape,
                                              mu,
                                              sigma)
        elif dist == 'uniform':
            return prng.randomize_uniform_core(keys, lat_shape)
        # elif dist not in valid_dists:
        #     raise ValueError(f"dist must be one of {valid_dists}; "
        #                      f"got {dist}.")

    def randomize_phi(self,
                      N_fields: int,
                      seed_or_key: int | jnp.ndarray | None = None,
                      randomize_keys: bool = True,
                      dist: str = 'normal',
                      mu: int | float | None = None,
                      sigma: int | float | None = None):
        """Re-initializes the phi-field.

        Generates N new keys, calls the JIT'd kernel,
        then mutates phi_x on the Python side. Also resizes the
        momentum field to match the new phi shape. This is a
        host-side operation.

        Parameters
        ----------
        N_fields : int
            Number of independent configurations.
        seed_or_key : int or jnp.ndarray or None
            Seed or PRNG key for reproducibility. If None and
            `randomize_keys=True`, fresh random keys are drawn.
        randomize_keys : bool
            Whether to generate new keys even when a seed is provided.
        dist : {'normal','uniform'}
            Distribution for initialization.
        mu, sigma : float or None
            Parameters for the normal distribution.

        Returns
        -------
        Phi4Lattice
            Self, with updated phi and momentum fields.
        """

        if not isinstance(N_fields, numbers.Integral) or N_fields <= 0:
            raise ValueError("N_fields must be positive integer.")
        if dist not in self.valid_dists:
            raise ValueError(f"dist must be one of {self.valid_dists}; "
                             f"got {dist}.")
        object.__setattr__(self, "phi_dist", dist)
        object.__setattr__(self, "phi_mu", mu)
        object.__setattr__(self, "phi_sigma", sigma)
        object.__setattr__(self, "n_keys", N_fields)
        object.__setattr__(self, "phi_seed", seed_or_key)

        master_key, keys = make_keys(N_fields, seed_or_key, randomize_keys)
        # update master_key and keys
        object.__setattr__(self, "phi_master_key", master_key)
        object.__setattr__(self, "phi_keys", keys)

        # selecting dist type
        rand_phi_xs = self._rand_field_core(keys,
                                            self.geom.lat_shape,
                                            self.phi_mu,
                                            self.phi_sigma,
                                            dist)
        object.__setattr__(self, 'phi_x', rand_phi_xs)

        # infer layout
        spatial_axes, shift = infer_layout(self.phi_x, self.geom.D)
        # update spatial_axes and shift
        object.__setattr__(self, 'spatial_axes',
                           tuple(int(x) for x in spatial_axes))
        object.__setattr__(self, 'shift', shift)

        # force momentum update to match new phi shape
        self.randomize_mom(seed_or_key=self.mom_seed,
                           randomize_keys=(self.mom_seed is None),
                           dist=self.mom_dist,
                           mu=self.mom_mu,
                           sigma=self.mom_sigma)
        return self

    # use for constant fields for simple magnetization expectations
    def constant_phi(self, constant):
        """Set the phi field to a uniform constant value.

        Parameters
        ----------
        constant : float
            Value to broadcast across the entire phi lattice.

        Returns
        -------
        Phi4Lattice
            Self, with updated phi.
        """
        if constant is None:
            raise ValueError("constant must be provided.")
        if not isinstance(constant, numbers.Real):
            raise TypeError("constant must be a real number.")
        phi_x = jnp.full_like(self.phi_x, constant, dtype=np.float64)
        object.__setattr__(self, 'phi_x', phi_x)
        return self

    # --- mom field methods --------------------
    def randomize_mom(self,
                      seed_or_key: int | jnp.ndarray | None = None,
                      randomize_keys=True,
                      dist='normal',
                      mu=None,
                      sigma=None):
        '''Randomize the momentum field.

        Asymmetric to randomize_phi since n_keys is not changed here.
        This is to force momentum sizes to match phi sizes. It will
        always match the shape of the phi field.

        Parameters
        ----------
        seed_or_key : int or jnp.ndarray or None
            PRNG seed or key.
        randomize_keys : bool
            Whether to generate fresh keys even when a seed is supplied.
        dist : {'normal','uniform'}
            Distribution to use.
        mu, sigma : float or None
            Parameters for the normal distribution.

        Returns
        -------
        Phi4Lattice
            Self, with updated momentum field.
        '''
        if dist not in self.valid_dists:
            raise ValueError(f"dist must be one of {self.valid_dists}; "
                             f"got {dist}.")
        object.__setattr__(self, "mom_dist", dist)
        object.__setattr__(self, "mom_mu", mu)
        object.__setattr__(self, "mom_sigma", sigma)
        object.__setattr__(self, "mom_seed", seed_or_key)

        if not isinstance(self.n_keys, numbers.Integral) or self.n_keys <= 0:
            raise ValueError("n_keys must be positive integer.")
        mom_master_key, mom_keys = make_keys(self.n_keys,
                                             seed_or_key,
                                             randomize_keys)
        object.__setattr__(self, "mom_master_key", mom_master_key)
        object.__setattr__(self, "mom_keys", mom_keys)
        mom_xs = self._rand_field_core(mom_keys,
                                       self.geom.lat_shape,
                                       mu=self.mom_mu,
                                       sigma=self.mom_sigma,
                                       dist=self.mom_dist)
        object.__setattr__(self, 'mom_x', mom_xs)
        return self

    def constant_momentum(self, constant):
        """Set the momentum field to a uniform constant value.

        Parameters
        ----------
        constant : float
            Constant value across the momentum field.

        Returns
        -------
        Phi4Lattice
            Self, with updated momentum field.
        """
        if constant is None:
            raise ValueError("constant must be provided.")
        if not isinstance(constant, numbers.Real):
            raise TypeError("constant must be a real number.")
        mom_x = jnp.full_like(self.mom_x, constant, dtype=np.float64)
        object.__setattr__(self, 'mom_x', mom_x)
        return self

    # ===========================================================
    # ------------- HMC field evolution -------------------------
    # returns in self for chaining
    # ===========================================================
    def run_HMC(self,
                cfg: params.HMCConfig,
                seed: int = None,
                randomize_keys: bool = False,
                measure_fns_dict: Dict[str, callable] = None
                ):
        """Run HMC trajectories to evolve the field.

        This updates phi_x and mom_x using the provided HMC
        configuration and returns self.

        Parameters
        ----------
        cfg : params.HMCConfig
            Configuration object specifying leapfrog steps, step size,
            trajectory count, and acceptance parameters.
        seed : int or None
            Seed used for PRNG key generation.
        randomize_keys : bool
            Whether to generate new keys.
        measure_fns_dict : dict[str, callable], optional
            Optional dictionary of measurement functions called during HMC
            evolution.

        Returns
        -------
        Phi4Lattice
            Self, with updated field, momentum, and trajectory history.
        """
        phi0 = self.phi_x
        mom0 = self.mom_x
        if not isinstance(cfg, params.HMCConfig):
            raise ValueError("cfg must be an instance of HMCConfig.")
        if seed is None:
            seed = cfg.seed
        if measure_fns_dict is not None:
            if not isinstance(measure_fns_dict, dict):
                raise TypeError("measure_fns_dict must be a "
                                "dict[str, Callable].")
            for name, fn in measure_fns_dict.items():
                if not callable(fn):
                    raise TypeError(f"measure_fns_dict[{name}] "
                                    "is not callable.")

        traj_master_key, traj_keys = prng._make_traj_keys(
                                                cfg.N_trajectories,
                                                seed_or_key=seed,
                                                randomize_keys=randomize_keys
                                                )
        object.__setattr__(self, 'hmc_master_key', traj_master_key)

        energy_fns = eng.make_phi4_energy_fns
        S_Fn, grad_S_Fn, H_kinetic_Fn = energy_fns(self.model,
                                                   self.geom,
                                                   self.shift,
                                                   self.spatial_axes)

        result = hmc.run_HMC_trajectories(phi0=phi0,
                                          mom0=mom0,
                                          traj_keys=traj_keys,
                                          cfg=cfg,
                                          S_Fn=S_Fn,
                                          grad_S_Fn=grad_S_Fn,
                                          H_kinetic_Fn=H_kinetic_Fn,
                                          measure_fns_dict=measure_fns_dict)
        (mom_final, phi_final), traj_outs_dict = result
        object.__setattr__(self, 'phi_x', phi_final)
        object.__setattr__(self, 'mom_x', mom_final)
        object.__setattr__(self, 'trajectory_history', traj_outs_dict)
        return self
