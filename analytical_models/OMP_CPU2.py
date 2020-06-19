"""
    OMP CPU2.py
        by Tim Müller
    
    This file implements the analytical model for the second variation of the
    OpenMP CPU-optimised implementations.
"""


import sequential



# BACKWARD PASS - OUTPUT
def t_bck_out(n_threads, pi, beta, L, P):
    return (sequential.t_bo_delta(pi, beta, P[L - 1]) / n_threads) + sequential.t_b_updbias(pi, beta, P[L - 1]) + sequential.t_b_updweights(pi, beta, P[L - 2], P[L - 1])



# BACKWARD PASS - HIDDEN
def t_bck_hid(n_threads, pi, beta, L, P):
    result = 0
    for l in range(1, L - 1):
        result += (sequential.t_bh_delta(pi, beta, P[l], P[l + 1]) / n_threads) + sequential.t_b_updbias(pi, beta, P[l]) + sequential.t_b_updweights(pi, beta, P[l - 1], P[l])
    return result



# Overwrite the predict function to take parallelism into account and call the special functions
def predict(sample_parameters, machine_parameters):
    n_threads = int(sample_parameters[0])
    L = int(sample_parameters[1]) + 2
    P = [int(sample_parameters[5])] + [int(sample_parameters[2])] * (L - 2) + [int(sample_parameters[6])]
    N = int(sample_parameters[3])
    S = int(sample_parameters[4])

    pi = machine_parameters[1][0] * 1000000000
    beta = machine_parameters[1][1] * 1000000000

    fwd_time = sequential.t_fwd(pi, beta, L, P) / n_threads
    bck_out_time = t_bck_out(n_threads, pi, beta, L, P)
    bck_hid_time = t_bck_hid(n_threads, pi, beta, L, P)
    updates_time = sequential.t_updates(pi, beta, L, P) / n_threads
    total_time = N * (S * (fwd_time + bck_out_time + bck_hid_time) + updates_time)

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]

