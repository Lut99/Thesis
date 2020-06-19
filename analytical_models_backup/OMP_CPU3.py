"""
    OMP CPU3.py
        by Tim Müller
    
    This file implements the analytical model for the third variation of the
    OpenMP CPU-optimised implementations.
"""


import sequential


# UPDATES
def t_updates(pi, beta, S, L, P):
    result = 0
    for l in range(L - 1):
        result += sequential.t_u_bias(pi, beta, P[l]) + sequential.t_u_weight(pi, beta, P[l], P[l + 1])
    return S * result



# Use what we can from sequential, but take different pi's and beta's into account
def predict(sample_parameters, machine_parameters):
    n_threads = int(sample_parameters[0])
    L = int(sample_parameters[1]) + 2
    P = [int(sample_parameters[5])] + [int(sample_parameters[2])] * (L - 2) + [int(sample_parameters[6])]
    N = int(sample_parameters[3])
    S = int(sample_parameters[4])

    pi = machine_parameters[n_threads][0] * 1000000000
    beta = machine_parameters[n_threads][1] * 1000000000

    fwd_time = sequential.t_fwd(pi, beta, L, P)
    bck_out_time = sequential.t_bck_out(pi, beta, L, P)
    bck_hid_time = sequential.t_bck_hid(pi, beta, L, P)
    updates_time = t_updates(pi, beta, S, L, P)
    total_time = N * (S * (fwd_time + bck_out_time + bck_hid_time) + updates_time)

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]
