#!/usr/bin/env python

import sys
sys.path.append('src')  # noqa
import argparse
import jax
import jax.numpy as jnp
from params import LatticeGeometry, Phi4Params, HMCConfig
from lattice import Phi4Lattice
from observables import magnetization, binder_cumulant
"""
Example script demonstrating HMC simulation of phi^4 theory on a 4D lattice.
can be run as: PYTHONPATH=. python scripts/example3.py
"""


def main():
    # --------------------------------------------------------------
    # 1. Set up lattice geometry and model parameters
    # --------------------------------------------------------------
    # Simple 4D lattice: 4^4 sites, with spacing a = 1 in each dimension.
    L_array = jnp.array([4, 4, 4, 4])
    a_array = jnp.ones_like(L_array)

    geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)
    model = Phi4Params(lam=1.0, kappa=0.1)

    # --------------------------------------------------------------
    # 2. HMC configuration
    # --------------------------------------------------------------
    cfg = HMCConfig(
        N_steps=10,
        eps=0.05,
        xi=0.2,
        integrator="omelyan",   # or "leapfrog"
        seed=123,
        N_trajectories=20,
        metropolis=True,
        record_H=False,
        verbose=False,
    )

    # --------------------------------------------------------------
    # 3. Build lattice and initialize fields
    # --------------------------------------------------------------
    # Start with 16 independent field configurations (batch = 16).
    lat = Phi4Lattice(model=model, geom=geom)
    lat.randomize_phi(
        N_fields=16,
        seed_or_key=0,          # deterministic batch init
        randomize_keys=False,
        dist="normal",
        mu=0.0,
        sigma=1.0,
    )

    print("Initial phi_x shape:", lat.phi_x.shape)
    print("Initial mom_x shape:", lat.mom_x.shape)

    # --------------------------------------------------------------
    # 4. Run HMC trajectories
    # --------------------------------------------------------------
    lat.run_HMC(
        cfg=cfg,
        seed=cfg.seed,          # seed for HMC trajectory keys
        randomize_keys=False,   # make it reproducible
        measure_fns_dict={"magnetization": lambda phi:
                          magnetization(phi, geom.D)}
    )

    phi_final = lat.phi_x
    print("Final phi_x shape:", phi_final.shape)

    # --------------------------------------------------------------
    # 5. Simple observables on the final configurations
    # --------------------------------------------------------------
    m = magnetization(phi_final, D=geom.D)
    print("Magnetization per configuration:", m)
    print("Average magnetization:", m.mean())

    U4 = binder_cumulant(phi_final, D=geom.D)
    print("Binder cumulant U4:", U4)

    # --------------------------------------------------------------
    # 6. Inspect acceptance statistics if present
    # --------------------------------------------------------------
    accept_mask = lat.trajectory_history.get("accept_mask", None)
    if accept_mask is not None:
        # accept_mask is boolean; average over all entries
        acc_rate = accept_mask.mean()
        print("Mean acceptance rate:", acc_rate)

    delta_H = lat.trajectory_history.get("delta_H", None)
    if delta_H is not None:
        print("Average ΔH over trajectories:", delta_H.mean())

    print("magnetization history:",
          lat.trajectory_history.get("magnetization", None))


if __name__ == "__main__":
    main()
