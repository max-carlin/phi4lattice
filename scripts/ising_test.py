import jax
import jax.numpy as jnp
from params import IsingParams, MetropMCConfig, LatticeGeometry
from lattice import IsingLattice
import sys
sys.path.append('src')

model = IsingParams(kappa=0.4, h=0.0)
L_array = jnp.array([4, 4], dtype=int)
a_array = jnp.array([1.0, 1.0])
geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)

cfg = MetropMCConfig(N_steps=5, seed=0)

lat = IsingLattice(model=model,
                   geom=geom,
                   n_keys=1,
                   sigma_dist='all-up')

print("Initial sigma_x:", lat.sigma_x)
print("initial shape:", lat.sigma_x.shape)

lat.run_Metropolis_MC(cfg=cfg)

print("Final sigma_x:", lat.sigma_x)
print("final shape:", lat.sigma_x.shape)
print("Trajectory history keys:", lat.trajectory_history.keys())
print("delta_S shape:", lat.trajectory_history['delta_S'].shape)
print("accept_mask shape:", lat.trajectory_history['accept_mask'].shape)


lat = IsingLattice(model=model,
                   geom=geom,
                   n_keys=1,
                   sigma_dist='uniform')

print("Initial sigma_x:", lat.sigma_x)
print("initial shape:", lat.sigma_x.shape)

lat.run_Metropolis_MC(cfg=cfg)
print("Final sigma_x:", lat.sigma_x)
print("final shape:", lat.sigma_x.shape)
print("Trajectory history keys:", lat.trajectory_history.keys())
print("delta_S shape:", lat.trajectory_history['delta_S'].shape)
print("accept_mask shape:", lat.trajectory_history['accept_mask'].shape)
