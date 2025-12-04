"""Run HMC simulation for phi^4 lattice on a 4D lattice.

This script runs a Hybrid Monte Carlo (HMC) simulation on a 4x4x4x4 lattice.
It constructs the lattices, initializes the field, and evolves them using
an HMC integrator (leapfrog or omelyan). It exposes all simulation parameters,
except for the lattice geometry, through command-line arguments.
"""
import sys
sys.path.append('src')  # noqa
import argparse
import jax
import jax.numpy as jnp
from params import LatticeGeometry, Phi4Params, HMCConfig
from lattice import Phi4Lattice
from observables import magnetization, binder_cumulant


def main(lam=1.0, kappa=0.1, N_steps=10, eps=0.05, xi=0.2,
         integrator="omelyan", seed=123, N_trajectories=20,
         metropolis=True, record_H=False, verbose=False,
         N_fields=16, seed_or_key=0, randomize_keys=False,
         mu=0.0, sigma=1.0):
    # Begin by constructing the lattice parameters
    # Simple 4D lattice: 4^4 sites, with spacing a = 1 in each dimension.
    L_array = jnp.array([4, 4, 4, 4])
    a_array = jnp.ones_like(L_array)

    geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)
    model = Phi4Params(lam, kappa)

    # Configure the HMC parameters
    cfg = HMCConfig(
        N_steps=N_steps,
        eps=eps,
        xi=xi,
        integrator=integrator,   # or "leapfrog"
        seed=seed,
        N_trajectories=N_trajectories,
        metropolis=metropolis,
        record_H=record_H,
        verbose=verbose,
    )

    # Now build the lattice and initialize fields.
    # Start with 16 independent field configurations (batch = 16).
    lat = Phi4Lattice(model=model, geom=geom)
    lat.randomize_phi(
        N_fields=N_fields,
        seed_or_key=seed_or_key,          # deterministic batch init
        randomize_keys=randomize_keys,
        dist="normal",
        mu=mu,
        sigma=sigma,
    )

    print("Initial phi_x shape:", lat.phi_x.shape)
    print("Initial mom_x shape:", lat.mom_x.shape)

    # Run HMC evolution
    lat.run_HMC(
        cfg=cfg,
        seed=cfg.seed,          # seed for HMC trajectory keys
        randomize_keys=False,   # make it reproducible
        measure_fns_dict=None
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Runs HMC simulation of phi^4 theory on a 4D lattice."),
        prog='run_hmc'
    )

    parser.add_argument('--lam',
                        type=float,
                        help='Coupling constant.',
                        default=1.0,
                        required=False)
    parser.add_argument('--kappa',
                        type=float,
                        help='Hopping parameter.',
                        default=0.1,
                        required=False)
    parser.add_argument('--N_steps',
                        type=int,
                        help='Number of integrator steps per trajectory.',
                        default=10,
                        required=False)
    parser.add_argument('--eps',
                        type=float,
                        help='Integrator step size.',
                        default=0.05,
                        required=False)
    parser.add_argument('--xi',
                        type=float,
                        help='Omelyan integrator parameter.',
                        default=0.2,
                        required=False)
    parser.add_argument('--integrator',
                        type=str,
                        help='Numerical integration method to use.',
                        default='omelyan',
                        required=False)
    parser.add_argument('--seed',
                        type=int,
                        help='Seed for random momentum metropolis test',
                        default=123,
                        required=False)
    parser.add_argument('--N_trajectories',
                        type=int,
                        help='Number of HMC trajectories to run.',
                        default=20,
                        required=False)
    parser.add_argument('--metropolis',
                        type=bool,
                        help='When True, uses metropolis acceptance test.',
                        default=True,
                        required=False)
    parser.add_argument('--record_H',
                        type=bool,
                        help='When True, records hamiltonian at each step.',
                        default=False,
                        required=False)
    parser.add_argument('--verbose',
                        type=bool,
                        help='When True, prints progress information.',
                        default=False,
                        required=False)
    parser.add_argument('--N_fields',
                        type=int,
                        help='Number of keys to make when generating.',
                        default=16,
                        required=False)
    parser.add_argument('--seed_or_key',
                        type=int,
                        help='Seed for PRNG.',
                        default=0,
                        required=False)
    parser.add_argument('--randomize_keys',
                        type=bool,
                        help='True if seed is None.',
                        default=False,
                        required=False)
    parser.add_argument('--mu',
                        type=float,
                        help='Mean for PRNG.',
                        default=0.0,
                        required=False)
    parser.add_argument('--sigma',
                        type=float,
                        help='Standard deviation for PRNG.',
                        default=1.0,
                        required=False)

    args = parser.parse_args()

    main(args.lam, args.kappa, args.N_steps, args.eps, args.xi,
         args.integrator, args.seed, args.N_trajectories,
         args.metropolis, args.record_H, args.verbose,
         args.N_fields, args.seed_or_key, args.randomize_keys,
         args.mu, args.sigma)
