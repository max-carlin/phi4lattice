import jax.numpy as jnp
import jax
from jax import jit  # use to translate to machine code for speed
from jax import random
from jax import lax
# from jax._src import dtypes/
from functools import partial
import numpy as np
# import pandas as pd
from dataclasses import dataclass, field, replace
# import matplotlib.pyplot as plt
from typing import Dict
jax.config.update("jax_enable_x64", True)  # 64 bit"""


@dataclass(frozen=True)
class Phi4Lattice:
    '''
    Lattice constructor draws form a normal
    distribution with mean mu and standard deviation sigma.
    '''
    # ---
    # defining param defaults and types of user-supplied values
    a_array:                   jnp.ndarray
    L_array:                   jnp.ndarray
    kappa:                     float
    lam:                       float
    mu:                        float = 0.0
    sigma:                     float = 0.1

    seed:                      int = 0
    n_keys:                    int = 1

    mom_seed:                  int = 1

    # --- Sec 1
    # derived fields/arrays
    # field geometry
    # # field(init=False) excludes as parameter to init method
    # # allows auto-gen of dataclass helpers like __repr__ (print of object)
    # # and coparative methods like __eq__ which might be useful later

    # lattice dim
    D:                         int = field(init=False)
    # lattice vol (rectangular)
    V:                         int = field(init=False)
    lat_shape:                 jnp.ndarray = field(init=False)

    # phi4 sampling
    master_key:                jnp.ndarray = field(init=False)
    keys:                      jnp.ndarray = field(init=False)
    rng:                       jnp.ndarray = field(init=False)
    phi_x:                     jnp.ndarray = field(init=False)

    # --- Sec 2
    mom_x:                     jnp.ndarray = field(init=False)
    mom_master_key:            jnp.ndarray = field(init=False)
    mom_keys:                  jnp.ndarray = field(init=False)
    spatial_axes:              tuple = field(init=False)
    shift:                     int = field(init=False)
    H_history:                 jnp.ndarray = field(init=False)
    measure_history:           Dict[str,
                                    jnp.ndarray] = field(init=False,
                                                         default_factory=dict)

    # set in post-init
    # action: callable = field(init=False)
    # grad_S: callable = field(init=False)

    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#
    # methods
    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#

    def __post_init__(self):
        '''
        initialization of geometric and field quantities (single draw of phi4)
        '''
        # --- Sec1
        # geom
        D = len(self.L_array)
        V = jnp.prod(self.L_array)
        lat_shape = tuple((self.L_array//self.a_array).tolist())

        # frozen = True -> set objects
        #   can't be reassigned
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
        rng = partial(random.normal,
                      shape=self.lat_shape,
                      dtype=jnp.float64)
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

        spatial_axes = tuple(range(self.phi_x.ndim - self.D,
                                   self.phi_x.ndim))
        object.__setattr__(self, 'spatial_axes',
                           tuple(int(x) for x in spatial_axes))
        shift = self.phi_x.ndim - self.D
        object.__setattr__(self, 'shift', shift)

        object.__setattr__(self, 'H_history', None)

        # # using modularity of the action to facilitate
        # # class flexibility for different potentials
        # action_fns = {
        #               'phi4': self._phi4_action,
        #               'harmonic': self._harmonic_action
        #               }

    # ----------------------------------------------------------------------------#
    # computation tools
    # ----------------------------------------------------------------------------#

    def make_keys(self, N, s=0, randomize_keys=True):
        '''
        prepare keys for sampling, return array of keys
        randomizes seed by default
        if randomize = False, seed is not randomized
        for traceability/reproducibility
        returns array of N jax PRNGkeys
        has to be host side because of np.random.randint
        '''
        # if seed is prng key, just split it
        if isinstance(s, jnp.ndarray) and s.shape == (2,):
            master = s
        # if seed is number, create prng key
        else:
            seed = int(s)  # ensures integer type
            if randomize_keys:
                seed = np.random.randint(0, N * 10**6)
            master = random.PRNGKey(seed)

        subkeys = random.split(master, N)
        return master, subkeys

    @staticmethod
    @partial(jax.jit,
             static_argnums=1)
    # issue with lat_shape is same as D in _magnetization_core
    # holding static on lat_shape/D seems to fix
    def _randomize_core(keys,
                        lat_shape,
                        mu,
                        sigma):
        """
        Pure JIT’d kernel
        given N keys, draws N phi-fields.
        lat_shape is static.
        """
        # vectorized normal draws
        rng = partial(random.normal,
                      shape=lat_shape,
                      dtype=jnp.float64)
        return jax.vmap(rng)(keys)

    @staticmethod
    @partial(jax.jit, static_argnums=1)
    # trying to reproduce fig 2.2
    def _randomize_uniform_core(keys,
                                lat_shape):
        rng = partial(random.uniform,
                      shape=lat_shape,
                      dtype=jnp.float64,
                      minval=0.0,
                      maxval=1.0)
        return jax.vmap(rng)(keys)

    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def _rand_phi_core(keys, lat_shape, mu, sigma):
        return mu + sigma * Phi4Lattice._randomize_core(keys,
                                                        lat_shape,
                                                        mu,
                                                        sigma)

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
        master_key, keys = self.make_keys(N, s, randomize_keys)
        object.__setattr__(self, "master_key", master_key)
        object.__setattr__(self, "keys", keys)

        # selecting dist type
        if dist == 'normal':
            rand_phi_xs = self._rand_phi_core(keys,
                                              self.lat_shape,
                                              self.mu,
                                              self.sigma)
        elif dist == 'uniform':
            rand_phi_xs = self._randomize_uniform_core(keys,
                                                       self.lat_shape)

        object.__setattr__(self, 'phi_x', rand_phi_xs)

        # determine if phi_x is singular or batched
        #   if phi_x.ndim == self.D then spatial axes are (0,...,D-1)
        #   if phi_x.ndim == self.D+1 then spatial axes are (1,...,D)
        #     N, the number of field configs, becomes the dimension 0

        # spatial will give tup(rang(0,3))
        #      = (0,1,2) for single 3D field
        #   or = (1,2,3) for batched 3D fields
        spatial_axes = tuple(range(self.phi_x.ndim - self.D, self.phi_x.ndim))
        # shift will = 0 for single field; = 1 for batch
        shift = self.phi_x.ndim - self.D

        object.__setattr__(self, 'spatial_axes',
                           tuple(int(x) for x in spatial_axes))
        object.__setattr__(self, 'shift', shift)
        return self

    def randomize_mom(self, N, s=1, randomize_keys=True):
        mom_master_key, mom_keys = self.make_keys(N,
                                                  s,
                                                  randomize_keys)
        object.__setattr__(self, "mom_master_key",
                           mom_master_key)
        object.__setattr__(self, "mom_keys", mom_keys)
        mom_xs = self._randomize_core(mom_keys,
                                      self.lat_shape,
                                      mu=0, sigma=1)
        object.__setattr__(self, 'mom_x', mom_xs)
        return self

    # ----------------------------------------------------------------------------#
    # action & grad(S)
    # ----------------------------------------------------------------------------#
    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4, 5))
    def _action_core(phi_x,
                     lam, kappa,
                     D,
                     shift, spatial_axes):
        '''
        Pure JIT’d kernel
        calculate and return action and kinetic energy

        phi_x:
          either single field, shape = (phi_0, ...,phi_{D-1})
          or array of fields, shape = (N, phi_0, ...,phi_{D-1})

        Returns:
          S: action
          K: kinetic energy
          W: interaction term
            w/ shape = (N,) if phi_x is array of
            fields or scalar if single field
        '''
        # Eq 1.1:
        #   S += -2 κ φ_x ∑_μ φ_{x+μ}  +  φ_x^2  +  λ(φ_x^2-1)^2
        K = 0
        for mu in range(D):
            ax = mu + shift
            # +mu -> +1; -mu -> -1,
            # no need for factor of 2 in action/kinetic term
            K += (phi_x * (jnp.roll(phi_x, 1, axis=ax)
                  + jnp.roll(phi_x, -1, axis=ax))).sum(axis=spatial_axes)

        K *= - kappa  # total kinetic
        # total potential
        U = (phi_x**2
             + lam * (phi_x**2 - 1.0)**2).sum(axis=spatial_axes)

        W = -K/kappa
        S = K + U
        return S, K, W

    def action_kinetic_W(self):
        S, K, W = self._action_core(self.phi_x, self.lam,
                                    self.kappa, self.D,
                                    self.shift, self.spatial_axes)
        return S, K, W

    def action(self):
        S, _, _ = self.action_kinetic_W()
        return S

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
    def _grad_action_core(phi_x,
                          lam,
                          kappa,
                          D,
                          shift,
                          spatial_axes):
        # total_action returns a scalar for jax.grad
        def total_action(phi):
            S_vals, _, _ = Phi4Lattice._action_core(phi,
                                                    lam, kappa,
                                                    D, shift,
                                                    spatial_axes)
            return jnp.sum(S_vals)
        # should compute grad(S) for both singular
        # or batched configs w/out axis error
        # that occured from other method
        return jax.grad(total_action)(phi_x)

    def grad_action(self, phi_x=None):
        if phi_x is None:
            phi_x = self.phi_x
        return self._grad_action_core(phi_x,
                                      self.lam, self.kappa,
                                      self.D, self.shift,
                                      self.spatial_axes)

    # ----------------------------------------------------------------------------#
    # magnetization & binder cummulant
    # ----------------------------------------------------------------------------#
    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def _magnetization_core(phi_x, D):
        '''
        Pure JIT’d kernel
        returns array of magnetizations for each field configuration in phi_x
        '''
        m_array = phi_x.sum(axis=tuple(range(1, D + 1)))
        return m_array

    def magnetization(self):
        '''
        returns array of magnetizations for each field configuration in phi_x
        '''
        return self._magnetization_core(self.phi_x, self.D)

    def binder_cumulant(self):
        """
        returns Binder cumulant, but only for batched fields
        """
        m = self.magnetization()
        m4_ave = (m**4).mean()
        m2_ave = (m**2).mean()
        return m4_ave / (m2_ave**2)

    # hamiltonian
    # ------------------------------------------------------------------#
    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def _hamiltonian_kinetic_core(mom_x, spatial_axes):
        # 1/2∑_x p_x²
        return (0.5*(mom_x**2).sum(axis=spatial_axes))

    def hamiltonian(self):
        S, K, W = self.action_kinetic_W()
        mom_term = self._hamiltonian_kinetic_core(self.mom_x,
                                                  self.spatial_axes)
        return mom_term + S

    # --------------------------------------------------------------------#
    # integrators
    # --------------------------------------------------------------------#

    # leap -----------------------------------------
    @staticmethod
    @partial(jax.jit, static_argnums=range(2, 10))
    # static: N_steps...spatial
    def _leapfrog_core_scan(mom_x0, phi_x0,
                            eps,
                            N_steps,
                            lam, kappa, D,
                            shift, spatial_axes,
                            record_H):
        # compute initial H if history is desired
        if record_H:
            S0, _, _ = Phi4Lattice._action_core(phi_x0,
                                                lam,
                                                kappa,
                                                D,
                                                shift,
                                                spatial_axes)
            H0 = Phi4Lattice._hamiltonian_kinetic_core(mom_x0,
                                                       spatial_axes) + S0

        def leap_step(state, _):
            mom_x_p, phi_x_p = state

            # I1 first half step; phi updates
            phi_x_p = phi_x_p + eps / 2 * mom_x_p
            # I2 pi updates whole step
            grad_s = Phi4Lattice._grad_action_core(phi_x_p,
                                                   lam,
                                                   kappa,
                                                   D,
                                                   shift,
                                                   spatial_axes)
            # grad_s = Phi4Lattice._grad_action_core(phi_x_p, lam, kappa, D)
            mom_x_p = mom_x_p - eps * grad_s
            # I1 second half step; phi updates again
            phi_x_p = phi_x_p + eps / 2 * mom_x_p

            # compute updated H after step
            if record_H:
                S_p, _, _ = Phi4Lattice._action_core(phi_x_p,
                                                     lam, kappa,
                                                     D,
                                                     shift,
                                                     spatial_axes)
                H_p = Phi4Lattice._hamiltonian_kinetic_core(mom_x_p,
                                                            spatial_axes) + S_p

                return (mom_x_p, phi_x_p), H_p
            return (mom_x_p, phi_x_p), None

        # run leap_step for N_steps
        (mom_fx, phi_fx), H_hist = lax.scan(leap_step,
                                            (mom_x0, phi_x0),
                                            xs=None,
                                            length=N_steps)

        # concat initil H0
        if record_H:
            H_hist = jnp.concatenate((H0[None], H_hist), axis=0)
            return mom_fx, phi_fx, H_hist
        return mom_fx, phi_fx

    def leap_frog_integrator(self,
                             N_steps,
                             eps,
                             N_trajectories=1,
                             record_H=False):
        def leap_trajectory(state, _):
            mom, phi = state
            output = self._leapfrog_core_scan(mom,
                                              phi,
                                              eps,
                                              N_steps,
                                              self.lam,
                                              self.kappa,
                                              self.D,
                                              self.shift,
                                              self.spatial_axes,
                                              record_H)
            if record_H:
                mom, phi, H = output
                return (mom, phi), H
            mom, phi = output
            return (mom, phi), None

        (mom_f, phi_f), H_hist = lax.scan(leap_trajectory,
                                          (self.mom_x, self.phi_x),
                                          xs=None,
                                          length=N_trajectories)
        # store final fields
        object.__setattr__(self, "mom_x", mom_f)
        object.__setattr__(self, "phi_x", phi_f)

        # store final history
        if record_H:
            object.__setattr__(self, "H_history", H_hist)
        return self

    # omelyan -----------------------------------------
    @staticmethod
    @partial(jax.jit,
             static_argnums=tuple(range(2, 11)))
    # N_steps through record_H static
    def _omelyan_core_scan(mom_x0, phi_x0,
                           N_steps,
                           lam, kappa, D,
                           shift, spatial_axes,
                           eps,
                           xi,
                           record_H):
        """
        One Omelyan trajectory of N_steps.
        If record_H -> also return H history (shape (N_steps+1, batch))
        """
        # pre-compute initial energy if Hamiltonian history is desired
        if record_H:
            S0, _, _ = Phi4Lattice._action_core(phi_x0,
                                                lam,
                                                kappa,
                                                D,
                                                shift,
                                                spatial_axes)
            H0 = Phi4Lattice._hamiltonian_kinetic_core(mom_x0,
                                                       spatial_axes) + S0

        def om_step(state, _):
            # scan expects an x input
            # along with carry/state even though xs=none
            mom_x_p, phi_x_p = state
            # I1(ξ eps)
            phi_x_p = phi_x_p + eps * xi * mom_x_p
            # I2(eps/2)
            grad_s = Phi4Lattice._grad_action_core(phi_x_p,
                                                   lam,
                                                   kappa,
                                                   D,
                                                   shift,
                                                   spatial_axes)
            # grad_s = Phi4Lattice._grad_action_core(phi_x_p, lam, kappa, D)
            mom_x_p = mom_x_p - eps / 2 * grad_s
            # I1((1-2ξ)eps)
            phi_x_p = phi_x_p + ((1 - 2 * xi) * eps)*mom_x_p
            # I2(eps/2)
            grad_s_p = Phi4Lattice._grad_action_core(phi_x_p,
                                                     lam,
                                                     kappa,
                                                     D,
                                                     shift,
                                                     spatial_axes)
            # grad_s_p = Phi4Lattice._grad_action_core(phi_x_p, lam, kappa, D)
            mom_x_p = mom_x_p - eps / 2 * grad_s_p
            # I1(ξ eps)
            phi_x_p = phi_x_p + eps * xi * mom_x_p

            if record_H:
                S_p, _, _ = Phi4Lattice._action_core(phi_x_p,
                                                     lam,
                                                     kappa,
                                                     D,
                                                     shift,
                                                     spatial_axes)
                H_p = Phi4Lattice._hamiltonian_kinetic_core(mom_x_p,
                                                            spatial_axes) + S_p
                return (mom_x_p, phi_x_p), H_p
            return (mom_x_p, phi_x_p), None

        # run om_step for N_steps; xs not needed (output only depends on state)
        # H_hist is the ys -- array of what the Hi values were after each step
        (mom_fx, phi_fx), H_hist = lax.scan(om_step,
                                            (mom_x0, phi_x0),
                                            xs=None,
                                            length=N_steps)

        if record_H:
            # include initial H val at position 0
            H_hist = jnp.concatenate((H0[None], H_hist), axis=0)
            return mom_fx, phi_fx, H_hist
        return mom_fx, phi_fx

    def omelyan_integrator(self, N_steps,
                           eps, xi,
                           N_trajectories=1,
                           record_H=False):
        def om_trajectory(state, _):
            mom, phi = state
            output = self._omelyan_core_scan(mom, phi,
                                             N_steps,
                                             self.lam,
                                             self.kappa,
                                             self.D,
                                             self.shift,
                                             self.spatial_axes,
                                             eps,
                                             xi,
                                             record_H)

            # output depends on if record_H, so it is unpacked differently
            if record_H:
                mom, phi, H = output
                return (mom, phi), H
            mom, phi = output
            return (mom, phi), None

        # use scan to loop again
        (mom_f, phi_f), H_hist = lax.scan(om_trajectory,
                                          (self.mom_x, self.phi_x),
                                          xs=None,
                                          length=N_trajectories)
        object.__setattr__(self, "mom_x", mom_f)
        object.__setattr__(self, "phi_x", phi_f)
        if record_H:
            object.__setattr__(self, "H_history", H_hist)

        return self

    # ----------------------------------------------------------------------------#
    # HMC update
    # ----------------------------------------------------------------------------#
    @staticmethod
    @jax.jit
    def _HMC_core(H_old, H_prime,
                  phi_old, phi_prime,
                  mom_old, mom_prime,
                  key):
        '''mask and update'''
        delta_H = H_prime - H_old  # H_final - H_initial
        # make acceptor mask
        r = random.uniform(key, shape=delta_H.shape)
        accept_mask = (delta_H < 0) | (r < jnp.exp(-delta_H))
        # reshape mask for batched
        #   will throw value error when batched if I don't
        #   ex) ValueError: Incompatible shapes for broadcasting:
        #       shapes=[(10,), (10, 4, 4, 4, 4), (10, 4, 4, 4, 4)]
        #   for batch of 10 configs
        # accept_mask.ndim = 0 for non batched, it's just a scalar
        #   mask = accept_mask
        # accept_mask.ndim = 1 for batched (basically the same as self.shift)
        # should give shape of (10, 1,1,1,1) for the above example
        mask = accept_mask.reshape(accept_mask.shape
                                   + (1,)*(phi_old.ndim-accept_mask.ndim))
        phi_accepted = jnp.where(mask, phi_prime, phi_old)
        mom_accepted = jnp.where(mask, mom_prime, mom_old)
        return mom_accepted, phi_accepted, mask, delta_H

    def HMC(self, N_steps, eps, xi,
            integrator='omelyan',
            s=0,
            N_trajectories=1,
            metropolis=True,
            record_H=False,
            verbose=False,
            *, measure_fns=None):
        '''
        Updates fields via HMC method, metroplis acceptor
        '''
        master_key = random.PRNGKey(np.random.randint(0, 10**6))
        # split master key into array of pairs
        traj_keys = random.split(master_key, 2*N_trajectories)
        # reshape for [(mom_key_1, r_key_1),...,(mom_key_N, r_key_N)]
        traj_keys = traj_keys.reshape((N_trajectories, 2, 2))

        # 4) molecular dynamics
        def MD_traj(state, key_pair):
            mom_old, phi_old = state
            # one key for momentum refresh, one for metrop
            mom_key, r_key = key_pair

            out: dict = {}

            # 1) refresh momentum field at the start of each trajectory
            mom_master_key, mom_keys = self.make_keys(mom_old.shape[0],
                                                      mom_key)
            mom_refreshed = self._randomize_core(mom_keys,
                                                 self.lat_shape,
                                                 mu=0,
                                                 sigma=1)

            if integrator == 'omelyan':
                output = self._omelyan_core_scan(mom_refreshed, phi_old,
                                                 N_steps,
                                                 self.lam,
                                                 self.kappa,
                                                 self.D,
                                                 self.shift,
                                                 self.spatial_axes,
                                                 eps,
                                                 xi,
                                                 record_H)
            if integrator == 'leap':
                output = self._leapfrog_core_scan(mom_refreshed,
                                                  phi_old,
                                                  eps,
                                                  N_steps,
                                                  self.lam,
                                                  self.kappa,
                                                  self.D,
                                                  self.shift,
                                                  self.spatial_axes,
                                                  record_H)

            mom_fx = output[0]
            phi_fx = output[1]
            if record_H:
                out['H_hist'] = output[2]

            if metropolis:
                # 5) calc H_f
                mom_t_prime = self._hamiltonian_kinetic_core(mom_fx,
                                                             self.spatial_axes)
                mom_term_prime = mom_t_prime
                S_prime, _, _ = self._action_core(phi_fx,
                                                  self.lam, self.kappa,
                                                  self.D, self.shift,
                                                  self.spatial_axes)
                H_prime = mom_term_prime + S_prime

                mom_t_old = self._hamiltonian_kinetic_core(mom_refreshed,
                                                           self.spatial_axes)
                mom_term_old = mom_t_old
                S_old, _, _ = self._action_core(phi_old,
                                                self.lam, self.kappa,
                                                self.D, self.shift,
                                                self.spatial_axes)
                H_old = mom_term_old + S_old

                # 3) current H val
                # 6) Metropolis test to update
                #    fields after each trajectory (tau)
                m_a, p_a, a_mask, d_H = self._HMC_core(H_old,
                                                       H_prime,
                                                       phi_old,
                                                       phi_fx,
                                                       mom_refreshed,
                                                       mom_fx,
                                                       r_key)
                mom_acc, phi_acc, accept_mask, delta_H = m_a, p_a, a_mask, d_H
                mom_fx = mom_acc  # overwrite final fields if accepted
                phi_fx = phi_acc
                out['traj_mom_keys'] = mom_keys
                out['traj_r_keys'] = r_key
                out['accept_mask'] = accept_mask
                out['delta_H'] = delta_H

            if measure_fns:
                for name, fn in measure_fns.items():
                    out[name] = fn(phi_fx)

            # if record_H:
            #   mom_fx, phi_fx, out
            #   # return (mom_fx, phi_fx), out
            return (mom_fx, phi_fx), out

        (mom_accepted,
         phi_accepted), out_dict = lax.scan(MD_traj,
                                            (self.mom_x,
                                             self.phi_x),
                                            xs=traj_keys,
                                            length=N_trajectories)

        object.__setattr__(self, 'mom_x', mom_accepted)
        object.__setattr__(self, 'phi_x', phi_accepted)

        if record_H or measure_fns or verbose:
            object.__setattr__(self, 'measure_history', out_dict)
        # if verbose:
        #   object.__setattr__(self, 'measure_history', out_dict)
        #   return self,
        return self

    # ----------------------------------------------------------------------------#
    # sanity checking tool
    # ----------------------------------------------------------------------------#

    # use for constant fields for simple magnetization expectations
    def _constant_field(self, constant):
        '''
        set all phi_x values to constant
        '''
        phi_x = jnp.full_like(self.phi_x, constant, dtype=np.float64)
        object.__setattr__(self, 'phi_x', phi_x)
        return self

    def _constant_momentum(self, constant):
        '''
        set all phi_x values to constant
        '''
        mom_x = jnp.full_like(self.mom_x, constant, dtype=np.float64)
        object.__setattr__(self, 'mom_x', mom_x)
        return self

    # ---------------------------------------------------------------------#
    # Parallelization tools
    # ---------------------------------------------------------------------#
    # @jax.jit
    def prime_factors(num):
        '''compute prime factors of num'''
        factors = []
        d = 2  # first prime
        while d * d <= num:
            while n % d == 0:  # force perfect division
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
            return factors

    @jax.jit
    def _all_splits(factors, n_unique_lengths):
        '''
        return combinatorial product of all unique factors
        used to find ways to split n_devices among lattice sub-blocks
          -> prod(factors) = n_devices, n_unique_lengths = N
          -> factors = [n1, n2, n3, ...]
          -> permutations = [(n_devices, 1, 1, ..., 1),
                            (1, n_devices, 1, ..., 1),
                            (1, 1, n_devices, ..., 1),
                            ...,
                            (1, 1, 1, ..., n_devices),
                            (n1, n2, n3, ..., nN),
                            (n1*n2, 1, n3, ..., nN),
                            (n1*n2*n3, 1, 1, ..., nN),
                            ...
                            ]

        for example:
          n_devices = 60 -> hand prime factorization
          and number of unique lattice lengths
          factors = [2,2,3,5], n_unique_lengths = 3
              -> want all triplet combination of products
          all_splits returns:
            [
            ( 60, 1, 1), ( 1, 60, 1), (1, 1, 60)]
            ( 12,  5, 1),  ( 12,  1, 5), ...,
            ( 4, 15, 1),  ( 4,  1, 15), ...,
            ( 10,  6, 1),  ( 1,  6, 10), ( 6, 10, 1), ( 6,  1, 10), ...,
            (20,  3, 1),  ( 1, 20, 3), ...,
            ...
            ]
        The triplets become proposals for sub-block scaling factors
          ex) the lattice is 3D, [120,60,30]
              the (60,1,1) split proposal would
              assign sub-blocks of the lattice
              of shape [2, 60, 30] to each device
              while (10,6,1) would assigns blocks
              like [12,10,30], a more cube-like sub-block
        '''
        N = n_unique_lengths
        ones = jnp.ones((N, N), dtype=int)
        identity = jnp.eye(N, dtype=int)
        splits = jnp.ones((1, N), dtype=int)
        for p in factors:
            M = ones - (p-1)*identity
            # gives matrix like
            # [(p,1,1,...,1), (1,p,1,...,1), ..., (1,1,1,...p)]

            # turn each element in splits into its own row
            # multiply each row in splits by each
            # row in M to get all possible products
            # then recover original shape & reassign to splits
            splits = (splits[:, :, None]*M[:, :, None]).reshape(-1, N)
            #   example: N=3, factors = [2,3,5]
            #     splits = (1,N) -> array([[1,1,1]])
            #     for p = 2
            #     choices -> Array([[2, 1, 1],
            #                       [1, 2, 1],
            #                       [1, 1, 2]], dtype=int64)

            #     choices[None,:,:] -> Array([[[2, 1, 1],
            #                                  [1, 2, 1],
            #                                  [1, 1, 2]]], dtype=int64)

            #     splits[:,None,:] -> Array([[[1, 1, 1]]], dtype=int64)

            #     (splits[:,:,None] * choices[None,:,:])
            #                  -> Array([[2, 1, 1],
            #                            [1, 2, 1],
            #                            [1, 1, 2]], dtype=int64)
            #     next, p = 3:
            #       choices[None,:,:] -> Array([[[3, 1, 1],
            #                                    [1, 3, 1],
            #                                    [1, 1, 3]]], dtype=int64)

            #       splits -> Array([[[2, 1, 1]],
            #                       [[1, 2, 1]],
            #                       [[1, 1, 2]]], dtype=int64)

            #       splits[:, :, None]*choices[None, :,:]
            #            -> Array([[6, 1, 1],
            #                      [2, 3, 1],
            #                      [1, 2, 3]],

            #                      [[3, 1, 2],
            #                       [1, 6, 1],
            #                       [1, 2, 3]],

            #                      [[3, 1, 2],
            #                       [1, 3, 2],
            #                       [1, 1, 6]]],dtype=int64)

            #     repeat step with 5 for final split
            #     Array([[30,  1,  1],
            #     [ 6,  5,  1],
            #     [ 6,  1,  5],
            #     [10,  3,  1],
            #     [ 2, 15,  1],
            #     [ 2,  3,  5],
            #     [10,  1,  3],
            #     [ 2,  5,  3],
            #     [ 2,  1, 15],
            #     [15,  2,  1],
            #     [ 3, 10,  1],
            #     [ 3,  2,  5],
            #     [ 5,  6,  1],
            #     [ 1, 30,  1],
            #     [ 1,  6,  5],
            #     [ 5,  2,  3],
            #     [ 1, 10,  3],
            #     [ 1,  2, 15],
            #     [15,  1,  2],
            #     [ 3,  5,  2],
            #     [ 3,  1, 10],
            #     [ 5,  3,  2],
            #     [ 1, 15,  2],
            #     [ 1,  3, 10],
            #     [ 5,  1,  6],
            #     [ 1,  5,  6],
            #     [ 1,   ,30]], dtype=int64)

        return splits

    def all_splits(primes, n_unique_lengths):
        '''
        return all ways the original number can be
        split n_unique_length ways
        as multiples of the prime factors
        used to find ways to split n_devices among lattice sub-blocks
          -> prod(factors) = n_devices, n_unique_lengths = N
          -> factors = [n1, n2, n3, ...]
          -> permutations = [(n_devices, 1, 1, ..., 1),
                            (1, n_devices, 1, ..., 1),
                            (1, 1, n_devices, ..., 1),
                            ...,
                            (1, 1, 1, ..., n_devices),
                            (n1, n2, n3, ..., nN),
                            (n1*n2, 1, n3, ..., nN),
                            (n1*n2*n3, 1, 1, ..., nN),
                            ...
                            ]

        for example:
          n_devices = 60 -> hand prime factorization
          and number of unique lattice lengths
          factors = [2,2,3,5], n_unique_lengths = 3
              -> want all triplet combination of products
          all_splits returns:
            Array([[60,  1,  1],
          [12,  5,  1],
          [12,  1,  5],
          [20,  3,  1],
          [ 4, 15,  1],
          [ 4,  3,  5],
          [20,  1,  3],
          [ 4,  5,  3],
          [ 4,  1, 15],
          [30,  2,  1],
          [ 6, 10,  1],
          [ 6,  2,  5],
          [10,  6,  1],
          [ 2, 30,  1],
          [ 2,  6,  5],
          [10,  2,  3],
          [ 2, 10,  3],
          [ 2,  2, 15],
          [30,  1,  2],
          [ 6,  5,  2],
          [ 6,  1, 10],
          [10,  3,  2],
          [ 2, 15,  2],
          [ 2,  3, 10],
          [10,  1,  6],
          [ 2,  5,  6],
          [ 2,  1, 30],
          [30,  2,  1],
          [ 6, 10,  1],
          [ 6,  2,  5],
          [10,  6,  1],
          [ 2, 30,  1],
          [ 2,  6,  5],
          [10,  2,  3],
          [ 2, 10,  3],
          [ 2,  2, 15],
          [15,  4,  1],
          [ 3, 20,  1],
          [ 3,  4,  5],
          [ 5, 12,  1],
          [ 1, 60,  1],
          [ 1, 12,  5],
          [ 5,  4,  3],
          [ 1, 20,  3],
          [ 1,  4, 15],
          [15,  2,  2],
          [ 3, 10,  2],
          [ 3,  2, 10],
          [ 5,  6,  2],
          [ 1, 30,  2],
          [ 1,  6, 10],
          [ 5,  2,  6],
          [ 1, 10,  6],
          [ 1,  2, 30],
          [30,  1,  2],
          [ 6,  5,  2],
          [ 6,  1, 10],
          [10,  3,  2],
          [ 2, 15,  2],
          [ 2,  3, 10],
          [10,  1,  6],
          [ 2,  5,  6],
          [ 2,  1, 30],
          [15,  2,  2],
          [ 3, 10,  2],
          [ 3,  2, 10],
          [ 5,  6,  2],
          [ 1, 30,  2],
          [ 1,  6, 10],
          [ 5,  2,  6],
          [ 1, 10,  6],
          [ 1,  2, 30],
          [15,  1,  4],
          [ 3,  5,  4],
          [ 3,  1, 20],
          [ 5,  3,  4],
          [ 1, 15,  4],
          [ 1,  3, 20],
          [ 5,  1, 12],
          [ 1,  5, 12],
          [ 1,  1, 60]], dtype=int64)

        The triplets become proposals for sub-block scaling factors
          ex) the lattice is 3D, [120,60,30]
              the (60,1,1) split proposal would assign
              sub-blocks of the lattice
              of shape [2, 60, 30] to each device
              while (10,6,1) would assigns blocks like [12,10,30],
              a more cube-like sub-block
        '''
        N = n_unique_lengths
        eye_N = jnp.eye(N, dtype=int)  # shape (N,N)
        ones = jnp.ones((N, N), dtype=int)  # shape (N,N)
        # create array of matrices containing possible choices of element
        # locations for each prime factor
        # ex) primes = array([2,3,5])
        #     p_choices = array([[2,1,1]],
        #                        [1,2,1],
        #                        [1,1,2]],
        #
        #                       [[3,1,1]],
        #                        [1,3,1],
        #                        [1,1,3]],
        #
        #                       [[5,1,1]],
        #                        [1,5,1],
        #                        [1,1,5]])

        # p_choices shape (P, N,N)
        p_choices = ones-eye_N+jnp.tensordot(primes, eye_N, axes=0)
        #   axis 0 → which sub-block (which prime factor)
        #   axis 1 → which row in that block
        #            (which row index is given to the factor)
        #   axis 2 → the factor columns

        P = p_choices.shape[0]  # number of prime factors; len(primes)

        # all ways to pick one row from each block
        # to get splits, we need to find all combinations of p_choices
        # ex) [2,1,1]*[3,1,1]*[5,1,1] = [30,1,1] is one choice
        #     [1,2,1]*[3,1,1]*[5,1,1] = [15,2,1] is another

        # (N,)*P -> P-tuple of N
        # jnp.indices((N,)*P) -> NxNx....xN (P times) grid
        #   ex) if N,P=2 we get 2x2 grid of indices
        #
        #      jnp.indices((2,)*2)
        #      Array([[[0, 0],
        #              [1, 1]],
        #
        #              [[0, 1],
        #              [0, 1]]], dtype=int64)
        # then reshape and transpose
        # 2x2 example -> array([[0, 0, 1, 1],
        #                        [0, 1, 0, 1]])
        row_combos = jnp.indices((N,)*P).reshape(P, -1)
        idx_row = row_combos  # shape (P, N**P)
        #  for each block, pick row corresponding to idx_row (the combinations)

        # idx_block shape (P,1); rows of single elements 0,1,...,P-1
        idx_block = jnp.arange(P)[:, None]
        #   picks block 0, block 1, ... etc
        picked = p_choices[idx_block, idx_row, :]  # shape (P, N**P, N)
        # from choices, the parent array of sub-blocks
        # with primes on diagonals & ones elsewhere
        # pick block __, take rows ___, and all columns
        # in other words, the combinations of rows
        # 2x2 ex)
        #         idx_block = array([[0],
        #                            [1]])
        #         idx_row = array([[0, 0, 1, 1],
        #                         [0, 1, 0, 1]])
        #         idx_block gets broadcast to match idx_row's shape
        #                 ->   array([[0,0,0,0],
        #                            [1,1,1,1]])
        #         giving combinations:
        #                             c=0: (block 0,row 0), (block 1,row 0)
        #                             c=1: (block 0,row 0), (block 1,row 1)
        #                             c=2: (block 0,row 1), (block 1,row 0)
        #                             c=3: (block 0,row 1), (block 1,row 1)

        # splits shape (N**P, N);
        # compute the products to get the device splits
        splits = picked.prod(axis=0)
        return splits

    @jax.jit
    def _surface_cost(lat_shape_arr, splits_arr):
        '''
        compute volume to surface area ratios of shards
        '''
        # vol = jnp.prod(lat_shape_arr)
        sub_blocks = lat_shape_arr//splits_arr
        sub_vol = jnp.prod(sub_blocks, axis=1)  # calculate row wise volumes
        # vol = lx*ly*lz; area_xy = vol/lz
        areas = sub_vol/sub_blocks
        total_area = areas.sum(axis=1)
        return total_area

    def get_best_device_count(lat_shape, n_devices):
        '''
        finds maximum number of devices that can be used for
        sharding without using jagged shards.
        checks for dvisibility and backs off
        one device at a time until a match is found
        '''
        pass

    def optimal_spatial_shards(self, n_dev):
        '''
        method to subdivide lattices for
        parallelization on large lattice structures
        aims to minimize surface area of the subdivisions
        n_dev = # of devices available for lattice sharding
        '''
        # from collections import defaultdict

        lat_shape_arr = jnp.array(self.lat_shape, dtype=int)
        # 1) check for and group by lattice symmetries to reduce search
        unique_lengths, inv_idx, counts = jnp.unique(lat_shape_arr,
                                                     return_inverse=True,
                                                     return_counts=True)
        # used for sharding permutations
        n_unique_lengths = len(unique_lengths)

        # 2) factorize n_devices
        def _get_best_split(n_dev):
            '''
            down number of devices used shift
            until clean device split can be made
            '''
            # down shift number of devices used shift until
            # clean device split can be made
            for d in range(n_dev, 0, -1):
                factors = self.prime_factors(d)

                # 3) enumerate every n_unique_lengths-tuple of device count
                dev_splits = self.all_splits(factors, n_unique_lengths)
                # 4) account for multiplicity in lattice lengths
                is_int = jnp.mod(dev_splits ** (1 / counts), 1) == 0
                # mask to get only splits that have
                # integer # when multiplicity is accounted for
                row_int_mask = is_int.all(axis=1)
                dev_splits_true = dev_splits[row_int_mask].astype(int)
                # 4) filter dev_splits for tuples
                #    that divide lattice lengths cleanly
                #   compute boolean mask to find
                #   dev_splits that have no remainder
                #   when used as divisor for lattice lengths
                mask = (unique_lengths % dev_splits_true == 0).all(axis=1)
                if jnp.any(mask):
                    # eliminate rows of splits that aren't all True
                    valid_splits = dev_splits_true[mask]
                    # expand out to full size from symmetry/multiplicity
                    valid_splits_full = valid_splits[:, inv_idx]
                    surface_cost = self._surface_cost(lat_shape_arr,
                                                      valid_splits_full)
                    best_split = valid_splits_full[jnp.argmin(surface_cost)]
                    return best_split
        best_split = _get_best_split(n_dev)
        return best_split

    def shard_uniform(self):
        '''
        Return a new lattice with sharded fields
          use replace() rather than object.__setattr___
          because there can be issues with
          mutating objects in a frozen data class.
          May need/want to swap other instances of
          object.__setattr__ for objects that
          I've mutated after they're first initialized in
          the init or post_init
        tries to find most cube-like (uniform) sharding structure
        may leave some devices idle
        '''
        from jax.sharding import PartitionSpec as P
        # get batch size
        n_batch = self.phi_x.shape[0] if self.shift == 1 else 1
        # total number of available devices
        n_dev = jax.device_count()
        # original, unsharded fields
        phi = self.phi_x
        mom = self.mom_x

        # case 1: fewer devices than field configurations (or equal)
        #         then split along the batch only (no parallelization error)
        if n_dev <= n_batch:
            mesh_shape = (n_dev,)
            axis_names = ('batch',)

        # case 2: single configuration, split the lattice only
        elif n_batch == 1:
            spatial_devices = self.optimal_spatial_shards(n_dev)
            mesh_shape = tuple(int(x) for x in spatial_devices)
            # generate axis strings from lattice geometry
            axis_names = tuple(f'n_{i}' for i in range(self.D))

        # case 3: more devices than configurations,
        #         split both batch and lattice
        else:
            # compute number of devices per batch element
            n_spatial_dev = n_dev//n_batch
            spatial_devices = self.optimal_spatial_shards(n_spatial_dev)
            mesh_shape = (n_batch,) + tuple(int(x) for x in spatial_devices)
            axis_names = ('batch',)+tuple(f'n_{i}' for i in range(self.D))

        # construct the mesh and sharding automatically with jax
        mesh = jax.make_mesh(mesh_shape, axis_names)
        sharding = jax.sharding.NamedSharding(mesh, P(axis_names))
        phi_sharded = jax.device_put(phi, sharding)
        mom_sharded = jax.device_put(mom, sharding)
        object.__setattr__(self, "phi_x", phi_sharded)
        object.__setattr__(self, "mom_x", mom_sharded)
        return self
