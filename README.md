# phi4lattice

There's an example in the src folder. That example looks like it works for the current implementation.

Here's the general package structure I'm thinking:
 - `integrators` holds the leapfrog and omelyan integratos. Right now, I still have to implement the leapfrog case. I noticed that hmc only calls on the `_core_scan` functions for both integrators, so I only have those included right now. 
 - `action` holds all the action logic, magnetization, and binder cumulant
 - `hmc` holds `HMC_core` and `MD_trajectory`, to run hmc though we use the hmc funciton in the Phi4Lattice class so that we can chain like in the original implementation
 - `lattice.py` handles the core class
 - `params.py` holds the HMC params class - this is just a dataclass that allows us to pass the lattice geometry between seperate files
 - `prng.py` handles the prng keys 

 Right now, this works for the example I have included in `example.py`. One of the big things I was trying to change was all the nested functions to make testing easier and follow best practices. lax.scan makes this difficult. Passing parameters through to the scanned functions (params.py) worked well. Had to use a lambda function for the initial carry state: 
  
(mom_fx, phi_fx), H_hist = lax.scan(
        lambda s, _: om_step(s, _, params),
        (mom_x0, phi_x0),
        xs=None,
        length=params.N_steps,
    )

  I could not avoid a nested function for `MD_traj`, however. If `measured_fns` is in params.py I get a JAX error saying a dict is not hashable. So, I have one nested function call `one_traj` in HMC. 

  TO-DO
  -----
- Implement leapfrog (should be fairly quick and easy)
- Write tests
- Format / follow best practices
- implement some sort of workflow - i'm thinking I'll create some visualization stuff
