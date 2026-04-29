import sys
sys.path.append('src')
import jax
import jax.numpy as jnp
from params import IsingParams, MetropMCConfig, LatticeGeometry
from lattice import IsingLattice
import numpy as np
import csv
from pathlib import Path
import argparse
import time

model = IsingParams(kappa=0.44, h=0.0)
N_sweeps = 10**5
threshold = 1
minimum_consecutive =3
num_seeds = 1
loops_to_thermalize_dict = {}
# lattice_sizes = [8, 16, 32, 64]
lattice_sizes = [32]
# sigma_dist = ['uniform', 'all-up']
# sigma_dist = ['all-up']
sigma_dist = ['uniform']
outpath = "results/ising_burn_in_estimate.csv"


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

# start = time.perf_counter()
for dist in sigma_dist:
    print(f"Testing thermalization for sigma_dist = {dist}")
    for L in lattice_sizes:
        L_array = jnp.array([L, L], dtype=int)
        a_array = jnp.array([1.0, 1.0])
        geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)
        N_proposed_flips = geom.V

        for i in range(num_seeds):
            # reinitialize lattice for each seed to ensure
            # independent thermalization tests
            lat = IsingLattice(model=model,
                        geom=geom,
                        n_keys=1,
                        sigma_dist=dist)
            
            seed = np.random.randint(0, 10**5)

            cfg = MetropMCConfig(N_steps=N_proposed_flips,
                                seed=seed)
            lat.thermalize(cfg=cfg,
                        max_loops=N_sweeps,
                        threshold=threshold,
                        store_loop_history=False,
                        store_error_history=False,
                        minimum_consecutive=minimum_consecutive)
            lat.sigma_x.block_until_ready()
            # elapsed_time_sec = time.perf_counter() - start
            # row = {"sigma_dist": dist,
            #     "L": L,
            #     "thermalized": lat.thermalization_diagnostics['thermalized'],
            #     "loops_to_thermalize": lat.thermalization_diagnostics['n_loops'],
            #     "seed": seed}
            row = {"kappa": float(model.kappa),
                    "threshold": float(threshold),
                    "minimum_consecutive": int(minimum_consecutive),
                    "max_loops": int(N_sweeps),
                    "sigma_dist": dist,
                    "L": int(L),
                    "thermalized": bool(lat.thermalization_diagnostics["thermalized"]),
                    "loops_to_thermalize": int(lat.thermalization_diagnostics["n_loops"]),
                    "seed": int(seed),
}
                # "elapsed_time_sec": elapsed_time_sec}
            append_row_csv(row, outpath)
# print("total elapsed time (sec):", time.perf_counter() - start)