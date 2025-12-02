# phi4lattice
> A reproducible, high performance hybrid monte carlo (HMC) engine for $\phi^4$ theory. 

phi4lattice provides a lightweight framework for simulating scalar field theory on a lattice. The package implements a modular Hybrid Monte Carlo (HMC) workflow. The codebase and subsequent simulations are designed to support small, simple experiments as well as more serious exploratory simulations.

## Project Structure
```
.
├── env.yml            # dependencies/version control
├── scripts            # example usage
│   ├── example.py
│   └── run_hmc.py
├── src                # code library
│   ├── __init__.py
│   ├── energetics.py    # action, grad S, hamiltonian
│   ├── hmc.py           # HMC engine with MD
│   ├── integrators.py   # leapfrog, Omelyan
│   ├── lattice.py       # Phi4Lattice, fields
│   ├── layout_utils.py  # infers batching
│   ├── observables.py   # ex) magnetization
│   ├── params.py        # configuration dataclasses
│   ├── prng.py          # PRNG key helpers
│   └── test_helpers.py
└── tests
    ├── func_tests
    │   └── test_run_hmc.sh
    └── unit_tests
        ├── test_energetics.py
        ├── test_hmc.py
        ├── test_integrators.py
        ├── test_layout_utils.py
        ├── test_observables.py
        ├── test_params.py
        └── test_prng.py
```

## Installation

From source:

```sh
git clone https://github.com/max-carlin/phi4lattice.git
cd phi4lattice
```

## Usage example

```python
import jax.numpy as jnp
from src.params import LatticeGeometry, Phi4Params, HMCConfig
from src.lattice import Phi4Lattice
from src.observables import magnetization, binder_cumulant

# Simple 4D lattice: 4^4 sites, with spacing a = 1 in each dimension.
L_array = jnp.array([4, 4, 4, 4])
a_array = jnp.ones_like(L_array)

# Set up lattice geometry and model parameters
# Simple 4D lattice: 4^4 sites, with spacing a = 1 in each dimension.
geom = LatticeGeometry(spacing_arr=a_array, length_arr=L_array)
model = Phi4Params(lam=1.0, kappa=0.1)

# HMC Configuration
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

lat = Phi4Lattice(model=model, geom=geom)
lat.randomize_phi(
    N_fields=16,
    seed_or_key=0,          # deterministic batch init
    randomize_keys=False,
    dist="normal",
    mu=0.0,
    sigma=1.0,
)

lat.run_HMC(
    cfg=cfg,
    seed=cfg.seed,          # seed for HMC trajectory keys
    randomize_keys=False,   # make it reproducible
    measure_fns_dict=None,  # keep it simple for the example
)

phi_final = lat.phi_x
m = magnetization(phi_final, D=geom.D)
```


## Development setup

Clone the repository and set up the environment.

```sh
git clone https://github.com/max-carlin/phi4lattice.git
cd phi4lattice
micromamba create -f env.yml
micromamba activate phi4lattice
```