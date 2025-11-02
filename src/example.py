import jax.numpy as jnp
from .lattice import Phi4Lattice
import matplotlib.pyplot as plt
import numpy as np

D=3
L=4

a_arr = jnp.ones(D, dtype = int)
l_arr = jnp.ones(D, dtype = int)*L
kappa = 0.5
lam = 1.3282

lat = Phi4Lattice(a_array=a_arr, L_array= l_arr, kappa=kappa, lam=lam)


N_steps = 10
N_therm = 10**4
xi = 0.1931833
eps=0.085
measure_fns = {'magnetization': lambda phi: Phi4Lattice._magnetization_core(phi,lat.D), 
               "phi_snapshot" : lambda phi : phi}
lat.HMC(N_steps, eps= eps, xi = xi, N_trajectories = N_therm, measure_fns=measure_fns, integrator='leap')
print('done')




x = np.arange(lat.measure_history['magnetization'].shape[0])
y = jnp.abs(lat.measure_history['magnetization'])

plt.scatter(x,y)
plt.show()
