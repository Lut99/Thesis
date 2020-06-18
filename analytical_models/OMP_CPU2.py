"""
    OMP CPU2.py
        by Tim Müller
    
    This file implements the analytical model for the second variation of the
    OpenMP CPU-optimised implementations.
"""


import sequential



# BACKWARD PASS - OUTPUT
def t_bck_out(pi_1, beta_1, pi_C, beta_C, L, P):
    return sequential.t_bo_delta(pi_C, beta_C, P[L - 1]) + sequential.t_b_updbias(pi_1, beta_1, P[L - 1]) + sequential.t_b_updweights(pi_1, beta_1, P[L - 2], P[L - 1])



# BACKWARD PASS - HIDDEN
def t_bck_hid(pi_1, beta_1, pi_C, beta_C, L, P):
    result = 0
    for l in range(1, L - 1):
        result += sequential.t_bh_delta(pi_C, beta_C, P[l], P[l + 1]) + sequential.t_b_updbias(pi_1, beta_1, P[l]) + sequential.t_b_updweights(pi_1, beta_1, P[l - 1], P[l])
    return result



# Use what we can from sequential, but take different pi's and beta's into account
def predict(sample_parameters, machine_parameters):
    n_threads = int(sample_parameters[0])
    L = int(sample_parameters[1]) + 2
    P = [int(sample_parameters[5])] + [int(sample_parameters[2])] * (L - 2) + [int(sample_parameters[6])]
    N = int(sample_parameters[3])
    S = int(sample_parameters[4])

    pi_1 = machine_parameters[0][0] * 1000000000
    beta_1 = machine_parameters[0][1] * 1000000000
    pi_C = machine_parameters[n_threads][0] * 1000000000
    beta_C = machine_parameters[n_threads][1] * 1000000000

    fwd_time = sequential.t_fwd(pi_C, beta_C, L, P)
    bck_out_time = t_bck_out(pi_1, beta_1, pi_C, beta_C, L, P)
    bck_hid_time = t_bck_hid(pi_1, beta_1, pi_C, beta_C, L, P)
    updates_time = sequential.t_updates(pi_C, beta_C, L, P)
    total_time = N * (S * (fwd_time + bck_out_time + bck_hid_time) + updates_time)

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]

