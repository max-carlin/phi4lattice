#!/bin/bash
test -e ssshtest || curl -s -O https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

script=scripts/run_hmc.py

run successful_run python $script
assert_exit_code 0 

run invalid_integrator python $script --integrator 'fake_integrator'
assert_exit_code 1
assert_stderr 
assert_in_stderr "Invalid integrator:"

run invalid_num_steps python $script --N_steps -5
assert_exit_code 1
assert_stderr 
assert_in_stderr "N_steps must be positive int."

run invalid_eps python $script --eps -5
assert_exit_code 1
assert_stderr 
assert_in_stderr "eps must be positive."

run invalid_seed python $script --seed 5.01
assert_exit_code 2

run invalid_N_trajectories python $script --N_trajectories -2
assert_exit_code 1
assert_stderr 
assert_in_stderr "N_trajectories must be positive int."

run invalid_metropolis python $script --metropolis 'wrong'
assert_stdout