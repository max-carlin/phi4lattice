import jax.numpy as jnp
from src.lattice import Phi4Lattice
import matplotlib.pyplot as plt
import numpy as np


D = 3  # spatial dimensions
L = 4  # length of each side of the lattice

# make D dimmensional hyper-square lattice of side length L
a_arr = jnp.ones(D, dtype=int)  # lattice spacing array
l_arr = jnp.ones(D, dtype=int)*L  # lattice shape array
kappa = 0.5  # hopping parameter
lam = 1.3282  # coupling parameter

# create lattice object
lat = Phi4Lattice(a_array=a_arr, L_array=l_arr, kappa=kappa, lam=lam)

N_steps = 10  # steps per trajectory
N_therm = 10**4  # thermalization steps
xi = 0.1931833  # integration time step for omelyan integrator
eps = 0.085  # step size

# define measurement functions
measure_fns = {'magnetization': lambda phi:
               Phi4Lattice._magnetization_core(phi, lat.D),
               "phi_snapshot": lambda phi: phi}

# run HMC
lat.HMC(N_steps,
        eps=eps,
        xi=xi,
        N_trajectories=N_therm,
        measure_fns=measure_fns,
        integrator='leapfrog')
print('done')

# plot magnetization history
x = np.arange(lat.measure_history['magnetization'].shape[0])
y = jnp.abs(lat.measure_history['magnetization'])

plt.scatter(x, y)
plt.show()
