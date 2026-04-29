import sys
sys.path.append('src')
import jax
import jax.numpy as jnp
from params import HMCConfig, LatticeGeometry, Phi4Params
from lattice import Phi4Lattice
import numpy as np
from collections import defaultdict
import csv
from pathlib import Path


ising_kappa_values = [0.1, 0.2, 0.3, 0.44,
                      0.5, 0.6, 0.7, 0.8]
kappa_values = [x/2 for x in ising_kappa_values]
lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
eps2_values_L8 = [
    0.126,   # lambda = 0.1
    0.0884,  # lambda = 0.5
    0.0689,  # lambda = 1.0
    0.0495,  # lambda = 2.0
    0.0300,  # lambda = 5.0
    0.0140,  # lambda = 10.0
    0.0016,  # lambda = 100.0
]
eps2_values_L16 = [
    0.1079,  # lambda = 0.1
    0.0689,  # lambda = 0.5
    0.0495,  # lambda = 1.0
    0.0300,  # lambda = 2.0
    0.0205,  # lambda = 5.0
    0.0095,  # lambda = 10.0
    0.0010,  # lambda = 100.0
]

eps2_values_L32 = [
    0.0884,  # lambda = 0.1
    0.0495,  # lambda = 0.5
    0.0300,  # lambda = 1.0
    0.0205,  # lambda = 2.0
    0.0140,  # lambda = 5.0
    0.0065,  # lambda = 10.0
    0.00066, # lambda = 100.0
]
eps2_values_L64 = []

eps2_dict = {8: eps2_values_L8,
              16: eps2_values_L16,
            32: eps2_values_L32,
            64: eps2_values_L64}


# Lengths = [8, 16, 32, 64]
lengths = [8]
a_array = jnp.array([1.0, 1.0])

parameters_dict = defaultdict(list)

# for L in lengths:
#     for lam, eps2 in zip(lambda_values, eps2_dict[L]):
#         parameters_dict[L].append((lam, np.sqrt(eps2)))
lam_eps_dict = {}
for L in lengths:
    assert len(eps2_dict[L]) == len(lambda_values), f"Mismatch for L={L}"
    lam_eps_dict[L] = [(lam, float(np.sqrt(eps2)))
                       for lam, eps2 in zip(lambda_values, eps2_dict[L])]

batch_size = 1
num_configs = 5
phi_dists = ['uniform', 'all-up']


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

result_dict = defaultdict(list)
for L in lengths:
    print(f"Testing thermalization for L = {L}")
    for dist in phi_dists:
        print(f"phi_dist = {dist}")
        for kappa in kappa_values:
            print(f"kappa = {kappa}")
            for lam, eps in lam_eps_dict[L]:
                print(f"lambda = {lam}, eps = {eps}")
                for config_idx in range(num_configs):

                    model = Phi4Params(kappa=kappa, lam=lam)
                    L_array = jnp.array([L, L], dtype=int)
                    geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)
                    if dist == 'uniform':
                        lat = Phi4Lattice(model=model,
                                    geom=geom,
                                    n_keys=batch_size,
                                    phi_dist=dist,
                                    phi_seed=np.random.randint(0, 10**5))
                    elif dist == 'all-up':
                        lat = Phi4Lattice(model=model,
                                    geom=geom,
                                    n_keys=batch_size)
                        lat.constant_phi(constant=1.0)

                    cfg = HMCConfig(eps=eps,
                                    xi=0.1931833,
                                    integrator='omelyan',
                                    N_steps=round(1/eps),
                                    N_trajectories=10**2,
                                    metropolis=True,
                                    record_H=False,
                                    verbose=False,
                                    seed=np.random.randint(0, 10**5))
                    lat.thermalize(cfg=cfg,
                                max_loops=10**4,
                                threshold=1,
                                minimum_consecutive=3,
                                randomize_keys=False)
                    diagnostics = lat.thermalization_diagnostics

                    row = {"L": L,
                        "phi_dist": dist,
                        "kappa": float(kappa),
                        "lam": float(lam),
                        "cfg_eps": float(cfg.eps),
                        "thermalized": bool(diagnostics["thermalized"]),
                        "N_traj_to_thermalize": int(diagnostics["n_trajectories"]),
                        "phi_seed": lat.phi_seed,
                        "hmc_seed": cfg.seed,
                        }
                    append_row_csv(row, out_path="results/phi4_thermalization_estimate.csv")
                





# model = Phi4Params(kappa = 0.22, lam = 1.0)
# N_trajectories = 10
# threshold = 1
# minimum_consecutive = 3
# num_seeds = 1

# lattice_sizes = [8, 16, 32, 64]
# phi_dist = ["all-up", "random-uniform"]

# def append_row_csv(row: dict,
#                       out_path: str):
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     file_exists = out_path.exists()
#     with out_path.open('a', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))

#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(row)

# for dist in phi_dist:
#     print(f"Testing thermalization for phi_dist = {dist}")
#     for L in lattice_sizes:
#         L_array = jnp.array([L, L, L, L], dtype=int)
#         a_array = jnp.array([1.0, 1.0, 1.0, 1.0])
#         geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)

#         for i in range(num_seeds):
#             # reinitialize lattice for each seed to ensure
#             # independent thermalization tests
#             lat = IsingLattice(model=model,
#                         geom=geom,
#                         n_keys=1,
#                         sigma_dist=dist)
            
#             seed = np.random.randint(0, 10**5)

#             cfg = HMCConfig(
#                 N_steps=20,
#                 eps=0.02,
#                 xi=0.1931833,
#                 integrator="omelyan",   # or "leapfrog"
#                 seed=seed,
#                 N_trajectories=N_trajectories,
#                 metropolis=True,
#                 record_H=False,
#                 verbose=False,
#             )

#             lat.thermalize(cfg=cfg,
#                             seed=seed,
#                             randomize_keys=False,
#                             threshold=threshold,
#                             minimum_consecutive=minimum_consecutive)

#             burn_in_info = lat.thermalization_diagnostics
#             print(f"Burn-in info for L={L}, dist={dist}, seed={seed}: {burn_in_info}")

#             row = {"L": L, "phi_dist": dist, "seed": seed, **burn_in_info}
#             append_row_csv(row=row, out_path="results/ising_burn_in_estimate.csv")