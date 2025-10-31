import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True) #64 bit
from jax import jit #use to translate to machine code for speed
from jax import random
from jax import lax
from functools import partial
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Dict
from .prng import init_fields
from .hmc import MD_traj
from .params import HMCParams

@dataclass(frozen=True)
class Phi4Lattice: 
    '''Core class to handle and create our lattice. 

    - Handles geometry, initialization
    '''

    a_array: jnp.ndarray
    L_array: jnp.ndarray
    kappa: float
    lam: float
    mu: float=0.0
    sigma: float=0.1
    seed: int=0
    n_keys: int=1
    mom_seed: int=1

    # Field Geometry
    D: int=field(init=False)
    V: int=field(init=False)
    lat_shape: jnp.ndarray=field(init=False)


    phi_x: jnp.ndarray=field(init=False)
    mom_x: jnp.ndarray=field(init=False)
    spatial_axes: tuple=field(init=False)
    shift: int=field(init=False)


    def __post_init__(self):
        D = len(self.L_array)
        V = jnp.prod(self.L_array)
        lat_shape = tuple((self.L_array // self.a_array).tolist())
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "V", V)
        object.__setattr__(self, "lat_shape", lat_shape)

        
        
        phi_x, mom_x, spatial_axes, shift = init_fields(lat_shape, self.seed, self.mom_seed, self.n_keys, self.mu, self.sigma, D)
        object.__setattr__(self, "phi_x", phi_x)
        object.__setattr__(self, "mom_x", mom_x)
        object.__setattr__(self, "spatial_axes", spatial_axes)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "master_key", random.PRNGKey(self.seed))
        object.__setattr__(self, 'H_history', None)

    def _split_keys(self, n):
        keys = random.split(self.master_key, n + 1)
        master_key, subkeys = keys[0], keys[1:]
        object.__setattr__(self, "master_key", master_key)
        return subkeys

    def HMC(self, N_steps, 
            eps, xi, integrator='omelyan', 
            s=0, N_trajectories=1, metropolis=True,
            record_H=False, verbose=False,
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
        # measure_fns cant't be in params.py because because
        # a dict is not hashable to JAX
        def one_traj(state, key_pair): 
            return MD_traj(state, key_pair, params, measure_fns)

        # need lambda key because jax.scan only takes (carry, element) NOT params
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
        m_array = phi_x.sum(axis = tuple(range(1,D+1)))
        return m_array

        

        
