"""Evolve a phi^4 lattice field using Hybrid Monte Carlo (HMC).

This script performs a Hybrid Monte Carlo simulation of a phi^4
field on a 4D lattice. It constructs the lattice geometry, initializes
the field configurations, and evolves them using either a leapfrog or
Omelyan integrator. Observables such as magnetization and the Binder
cumulant are computed, and the field evolution is saved as a video
(animation) for visualization.

Parameters and configuration (except lattice geometry) are controlled
via the HMCConfig dataclass. The script is designed for experimentation
with step sizes, trajectory lengths, and integrator choices.
"""
# !/usr/bin/env python
import sys
sys.path.append('src')  # noqa
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from params import LatticeGeometry, Phi4Params, HMCConfig
from lattice import Phi4Lattice
from observables import magnetization, binder_cumulant


def main():
    # Begin by constructing the lattice parameters
    # Simple 4D lattice: 4^4 sites, with spacing a = 1 in each dimension.
    L_array = jnp.array([10, 10, 10, 10])
    a_array = jnp.ones_like(L_array)

    geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)
    model = Phi4Params(lam=1.3282, kappa=0.18)

    # Configure the HMC parameters
    cfg = HMCConfig(
        N_steps=20,
        eps=0.02,
        xi=0.1931833,
        integrator="omelyan",   # or "leapfrog"
        seed=123,
        N_trajectories=1000,
        metropolis=True,
        record_H=False,
        verbose=False,
    )

    # Now build the lattice and initialize fields.
    lat = Phi4Lattice(model=model, geom=geom)
    lat.randomize_phi(
        N_fields=1,
        seed_or_key=0,          # deterministic batch init
        randomize_keys=False,
        dist="normal",
        mu=0.0,
        sigma=1.0,
    )

    print("Initial phi_x shape:", lat.phi_x.shape)
    print("Initial mom_x shape:", lat.mom_x.shape)

    # Run HMC evolution
    lat.run_HMC(
        cfg=cfg,
        seed=cfg.seed,          # seed for HMC trajectory keys
        randomize_keys=False,   # make it reproducible
        measure_fns_dict={"magnetization": lambda phi:
                          magnetization(phi, geom.D),
                          "phi": lambda phi: phi}
    )

    phi_final = lat.phi_x
    print("Final phi_x shape:", phi_final.shape)

    # Save simple observables from the HMC process
    m = magnetization(phi_final, D=geom.D)
    print("Magnetization per configuration:", m)
    print("Average magnetization:", m.mean())

    U4 = binder_cumulant(phi_final, D=geom.D)
    print("Binder cumulant U4:", U4)

    # Inspect acceptance statistics to ensure viable integrator params
    accept_mask = lat.trajectory_history.get("accept_mask", None)
    if accept_mask is not None:
        # Accept_mask is boolean; average over all entries
        acc_rate = accept_mask.mean()
        print("Mean acceptance rate:", acc_rate)

    delta_H = lat.trajectory_history.get("delta_H", None)
    if delta_H is not None:
        print("Average ΔH over trajectories:", delta_H.mean())

    print("magnetization history:",
          lat.trajectory_history.get("magnetization", None).shape)

    phi_hist = lat.trajectory_history["phi"]
    print("phi history shape:", phi_hist.shape)

    # This code saves the evolution of the phi-field as a video
    fps = 60
    output_name = "phi_evolution_fast.mp4"

    writer = ani.FFMpegWriter(fps=fps)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(1.0)

    phi0 = phi_hist[0][0][:, :, 0, 0]
    im = ax.imshow(phi0, cmap='RdBu', animated=True)
    cb = fig.colorbar(im, ax=ax, label=r'$\phi$')
    ax.set_title("Φ-field evolution")
    ax.axis("off")
    i = 0
    with writer.saving(fig, output_name, dpi=120):

        for phi in phi_hist:
            i += 1
            phi_slice = phi[0][:, :, 0, 0]

            im.set_data(phi_slice)

            writer.grab_frame()
            if i % 100 == 0:
                print(f'Saving frame {i}')


if __name__ == "__main__":
    main()
