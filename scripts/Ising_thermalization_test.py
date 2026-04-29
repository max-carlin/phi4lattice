"""
Test script for thermalization diagnostics of the Ising lattice.
Assuming you start in enclosing phi4lattice_folder, run with:
PYTHONPATH=src python scripts/Ising_thermalization_test.py
Note: Ising model isnot yet optimized with jax based code
"""
# !/usr/bin/env python
import sys
sys.path.append('src')
import jax
import jax.numpy as jnp
from params import IsingParams, MetropMCConfig, LatticeGeometry
from lattice import IsingLattice
import numpy as np




model = IsingParams(kappa=0.4, h=0.0)
L_array = jnp.array([16, 16], dtype=int)
a_array = jnp.array([1.0, 1.0])
geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)

# 1 MC sweep is ~ V number of proposed flips,
# so we set N_steps to 10*V to ensure sufficient
# sampling for thermalization diagnostics
N_proposed_flips = geom.V
N_sweeps = 10**2
threshold = 1
seed = np.random.randint(0, 10000)  # random seed for thermalization test reproducibility
cfg = MetropMCConfig(N_steps=N_proposed_flips,
                     seed=seed)

lat = IsingLattice(model=model,
                   geom=geom,
                   n_keys=1,
                   sigma_dist='all-up')
# lat = IsingLattice(model=model,
#                    geom=geom,
#                    n_keys=1,
#                    sigma_dist='uniform')

print("Initial sigma_x:", lat.sigma_x)
print("initial shape:", lat.sigma_x.shape)

lat.thermalize(cfg=cfg,
               max_loops=N_sweeps,
               threshold=threshold,
               store_loop_history=True,
               store_error_history=True)
diagnostics = lat.thermalization_diagnostics
history = diagnostics['history']

print({"diagnostics keys": diagnostics.keys(),
       "thermalized": diagnostics['thermalized'],
       "n_loops": diagnostics['n_loops']})

print("Final sigma_x:", lat.sigma_x)
print("final shape:", lat.sigma_x.shape)
print("Trajectory history keys:", lat.trajectory_history.keys())
print("mag_error shape::", history['mag_error'].shape)
print("mag_error", history['mag_error'])
print("action_error", history['action_error'])


# lat = IsingLattice(model=model,
#                    geom=geom,
#                    n_keys=1,
#                    sigma_dist='uniform')

# print("Initial sigma_x:", lat.sigma_x)
# print("initial shape:", lat.sigma_x.shape)

# lat.thermalize(cfg=cfg,
#                max_loops=N_sweeps,
#                threshold=threshold)
# print(lat.thermalization_diagnostics)

N_sweeps = 10**2
num_seeds = 5
loops_to_thermalize_dict = {}
lattice_sizes = [8, 16, 32, 64]
sigma_dist = ['uniform', 'all-up']

for dist in sigma_dist:
    print(f"Testing thermalization for sigma_dist = {dist}")
    for L in lattice_sizes:
        L_array = jnp.array([L, L], dtype=int)
        a_array = jnp.array([1.0, 1.0])
        geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)
        N_proposed_flips = geom.V

        loops_to_thermalize = []
        for i in range(num_seeds):
            # reinitialize lattice for each seed to ensure
            # independent thermalization tests
            lat = IsingLattice(model=model,
                        geom=geom,
                        n_keys=1,
                        sigma_dist=dist)
            
            seed = np.random.randint(0, 10000)

            cfg = MetropMCConfig(N_steps=N_proposed_flips,
                                seed=seed)
            lat.thermalize(cfg=cfg,
                        max_loops=N_sweeps,
                        threshold=threshold,
                        store_loop_history=False,
                        store_error_history=False)
            loops_to_thermalize.append(lat.thermalization_diagnostics['n_loops'])
        loops_to_thermalize_dict[(dist, L)] = loops_to_thermalize
print("Loops to thermalize for each (sigma_dist, L):")
for key, value in loops_to_thermalize_dict.items():
    print(f"{key}: {value}")
print("Average loops to thermalize for each (sigma_dist, L):")
for key, value in loops_to_thermalize_dict.items():
    print(f"{key}: {np.mean(value)}")
