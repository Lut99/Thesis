"""
    OMP CPU1.py
        by Tim Müller
    
    This file implements the analytical model for the first variation of the
    OpenMP CPU-optimised implementations.
"""


import sequential


# Use the functions in sequential, but inplement our own predict to be able to
#   differ the pis and betas used
def predict(sample_parameters, machine_parameters):
    n_threads = int(sample_parameters[0])
    L = int(sample_parameters[1]) + 2
    P = [int(sample_parameters[5])] + [int(sample_parameters[2])] * (L - 2) + [int(sample_parameters[6])]
    N = int(sample_parameters[3])
    S = int(sample_parameters[4])

    pi_1 = machine_parameters[1][0] * 1000000000
    beta_1 = machine_parameters[1][1] * 1000000000
    pi_C = machine_parameters[n_threads][0] * 1000000000
    beta_C = machine_parameters[n_threads][1] * 1000000000

    fwd_time = sequential.t_fwd(pi_C, beta_C, L, P)
    bck_out_time = sequential.t_bck_out(pi_1, beta_1, L, P)
    bck_hid_time = sequential.t_bck_hid(pi_1, beta_1, L, P)
    updates_time = sequential.t_updates(pi_C, beta_C, L, P)
    total_time = N * (S * (fwd_time + bck_out_time + bck_hid_time) + updates_time)

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]
