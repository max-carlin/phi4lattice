import sys
sys.path.append('src')
import jax
import jax.numpy as jnp
from params import IsingParams, MetropMCConfig, LatticeGeometry
from lattice import IsingLattice
import numpy as np
import csv
from pathlib import Path
import observables as obs
import energetics as en

# kappa_values = [0.1, 0.2, 0.3, 0.4,
#                 0.44,
#                 0.5, 0.6, 0.7, 0.8]
# kappa_values = [0.42, 0.46, 0.9, 1.0, 1.1]
# kappa_values = [11.0, 2.75, 1.6, 1.1, 0.85, 0.69, 0.58, 0.5,
#                  0.44,
#                  0.4, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22]
# kappa_values = [1.1, 0.85, 0.69, 0.58, 0.5,
#                 0.47, 0.45, 0.44, 0.43, 0.41,
#                 0.4, 0.32, 0.30, 0.28, 0.26]
kappa_values = [0.36]
# N_sweeps = [246,458,917,1353]*10
# lattice_sizes = [8, 16, 32, 64]
N_sweeps = 1353*10
lattice_size = 64
# threshold = 1
# minimum_consecutive = 5

# sigma_dist = ['uniform', 'all-up']
sigma_dist = 'uniform'
outpath = "results/ising_observables_results.csv"

def append_row_csv(row: dict,
                      out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists()
    with out_path.open('a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

for kappa in kappa_values:
    print(f"Testing thermalization for kappa = {kappa}")
    model = IsingParams(kappa=kappa, h=0.0)
    L_array = jnp.array([lattice_size, lattice_size], dtype=int)
    a_array = jnp.array([1.0, 1.0])
    geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)
    N_proposed_flips = geom.V


    # reinitialize lattice for each seed to ensure
    # independent thermalization tests
    lat = IsingLattice(model=model,
                geom=geom,
                n_keys=1,
                sigma_dist=sigma_dist)
    
    burn_seed = np.random.randint(0, 10**5)

    cfg = MetropMCConfig(N_steps=N_sweeps*geom.V,
                        seed=burn_seed)
    # lat.thermalize(cfg=cfg,
    #             max_loops=N_sweeps,
    #             threshold=threshold,
    #             store_loop_history=False,
    #             store_error_history=False,
    #             minimum_consecutive=minimum_consecutive)

    lat.run_Metropolis_MC(cfg=cfg,
                        store_proposal_history = False,
                        thermalization_summary = False)

    
    # collect post-thermalization observables
    meas_seed = np.random.randint(0, 10**5)
    cfg = MetropMCConfig(N_steps=N_proposed_flips*10**4,
                        seed=meas_seed)
    
    measure_fns_dict = {"magnetization": lambda sigma_x: obs.magnetization(sigma_x,
                                                                           lat.spatial_axes,
                                                                           lat.geom.V),
                        "action": lambda sigma_x: en.ising_action_core(sigma_x,
                                                                                lat.model,
                                                                                lat.geom,
                                                                                lat.shift,
                                                                                lat.spatial_axes
                                                                                )[0]}
    lat.run_Metropolis_MC(cfg=cfg,
                        measure_fns_dict=measure_fns_dict,
                        store_proposal_history=False,
                        thermalization_summary=False
                            )
    # Ensure all computations are 
    # complete before accessing trajectory history
    lat.sigma_x.block_until_ready()

    m = lat.trajectory_history['magnetization']
    S = lat.trajectory_history['action']

    m_abs_mean = jnp.abs(m).mean()
    m_mean = m.mean()
    m2_mean = (m**2).mean()
    m4_mean = (m**4).mean()
    chi=geom.V*(m2_mean - m_mean**2)
    binder_cumulant = 1 - m4_mean/(3*m2_mean**2)
    S_mean = S.mean()
    S2_mean = (S**2).mean()
    C_per_Nk = (S2_mean - S_mean**2)/geom.V

    row = {"kappa": kappa,
           "L": lattice_size,
        #    'thermalized': lat.thermalization_diagnostics['thermalized'],
        #    "N_therm_sweeps": lat.thermalization_diagnostics['n_loops'],
           "N_meas_sweeps": 10**4,
           "burn_seed": burn_seed,
           "meas_seed": meas_seed,
           "magnetization": m_mean,
           "abs_magnetization": m_abs_mean,
           "magnetization_squared": m2_mean,
           "action": S_mean,
           "action_squared": S2_mean,
           "susceptibility": chi,
           "binder_cumulant": binder_cumulant,
           "C_per_Nk": C_per_Nk}
    append_row_csv(row, outpath)