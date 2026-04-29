import sys
sys.path.append('src')
import jax
import jax.numpy as jnp
from params import LatticeGeometry, Phi4Params, HMCConfig
from lattice import Phi4Lattice
import numpy as np
import csv
from pathlib import Path


# kappa_values = [1.1, 0.85, 0.69, 0.58, 0.5,
#                 0.47, 0.45, 0.44, 0.43, 0.41,
#                 0.4, 0.36, 0.32, 0.30, 0.28, 0.26]
# kappa_values = [0.85, 0.5, 0.44, 0.36, 0.28]
# kappa_values = [0.85, 0.44, 0.28]
kappa_values = [0.44]/2
# lattice_sizes = [8, 16, 32, 64]
lattice_sizes = [8, 16, 32,64]
# dist_type = ['uniform', 'all-up']
dist_type = 'uniform'
lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
# lambda_values = [0.1, 1, 10.0, 100.0]
batch_size = 10**2
N_trajectories = 10**2
N_trajectories_burn = 10**3

# eps_values = np.geomspace(0.01, np.sqrt(0.08), 10)
# eps2_log = np.geomspace(1e-4, 0.02, 7)
# eps2_lin = np.linspace(0.025, 0.08, 4)
# eps2_values = np.unique(np.concatenate([eps2_log, eps2_lin]))
# eps_values = np.sqrt(eps2_values)

# eps2_log = np.geomspace(1e-6, 0.2, 20)
# eps2_lin = np.linspace(1e-6, 0.4, 20)
# eps2_values = np.unique(np.concatenate([eps2_log, eps2_lin]))
# eps_values = np.sqrt(eps2_values)

eps2_geom = np.geomspace(1e-6, 0.03, 28)
eps2_lin  = np.linspace(0.03, 0.40, 20)
eps2_values = np.unique(np.concatenate([eps2_geom, eps2_lin]))
eps_values = np.sqrt(eps2_values)



def make_lattice(kappa, lam, L, dist):
    model = Phi4Params(kappa=kappa, lam=lam)
    L_array = jnp.array([L, L], dtype=int)
    a_array = jnp.array([1.0, 1.0])
    geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)

    lat = Phi4Lattice(model=model,
                geom=geom,
                n_keys=1,
                phi_dist=dist)
    # print("phi:", lat.phi_x)
    # print("mom:", lat.mom_x)
    return lat

def make_cfg(seed, N_trajectories, eps):
    cfg = HMCConfig(
        N_steps=round(1/eps),  # rounds to nearest integer
        eps=eps,
        xi=0.1931833,
        integrator="omelyan",   # or "leapfrog"
        seed=seed,
        N_trajectories=N_trajectories,
        metropolis=True,
        record_H=False,
        verbose=False,
    )
    return cfg

def eps_scan(kappa, lam, L, dist_type, eps):
    lat = make_lattice(kappa, lam, L, dist_type)

    lat.randomize_phi(N_fields = batch_size,
                      randomize_keys=True,
                      dist=dist_type)
    lat.randomize_mom(randomize_keys=True,
                      dist = 'normal')
    
    meas_cfg = make_cfg(seed=np.random.randint(0, 10**5),
                    N_trajectories=N_trajectories,
                    eps=eps)
    burn_cfg = make_cfg(seed=meas_cfg.seed,
                        N_trajectories=N_trajectories_burn,
                        eps=eps)
    # warm up the lattice with a long HMC trajectory to reach typical configurations
    lat.run_HMC(cfg=burn_cfg,
                randomize_keys=False)
    
    lat.run_HMC(cfg=meas_cfg,
                randomize_keys=False)
    accept_rate = float(lat.trajectory_history["accept_mask"].mean())
    delta_H_abs_mean = float(jnp.abs(lat.trajectory_history["delta_H"]).mean())
    delta_H_abs_std = float(jnp.abs(lat.trajectory_history["delta_H"]).std())

    out_dict = {"seed": meas_cfg.seed,
                "N_steps": meas_cfg.N_steps,
                "eps": meas_cfg.eps,
                "accept_rate": accept_rate,
                "delta_H_abs_mean": delta_H_abs_mean,
                "delta_H_abs_std": delta_H_abs_std}
    return out_dict

outpath = "results/phi4_acceptance_rates.csv"
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

for L in lattice_sizes:
    print(f"for L = {L}")
    for kappa in kappa_values:
        print(f"for kappa = {kappa}")
        for lam in lambda_values:
            print(f"for lambda = {lam}")
            for eps in eps_values:
                # print(f"for eps = {eps}")
                result = eps_scan(kappa,
                                            lam,
                                            L,
                                            dist_type,
                                            eps)
                
                row = {'L': L,
                        'kappa': kappa, 
                        'lambda': lam,
                        'eps_2': eps**2,
                        'eps': eps,
                        'acceptance_rate': result["accept_rate"],
                        'delta_H_abs_mean': result["delta_H_abs_mean"],
                        'delta_H_abs_std': result["delta_H_abs_std"],
                        'N_steps': result["N_steps"],
                        'N_trajectories': N_trajectories,
                        'batch_size': batch_size,
                        'seed': result["seed"],}
                append_row_csv(row, outpath)


