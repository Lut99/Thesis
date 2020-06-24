"""
    OMP CPU3.py
        by Tim Müller
    
    This file implements the analytical model for the third variation of the
    OpenMP CPU-optimised implementations.
"""


import sequential


# BACKWARD PASS - OUTPUT
def t_bck_out(pi, beta, L, P):
    return sequential.t_bo_delta(pi, beta, P[L - 1]) + t_b_updbias(beta, P[L - 1]) + t_b_updweights(pi, beta, P[L - 2], P[L - 1])

def t_b_updbias(beta, p):
    return (16 * p) / beta

def t_b_updweights(pi, beta, p, p1):
    if pi <= beta * (1 / 24):
        return (p * p1) / pi
    else:
        return (24 * p * p1) / beta


# BACKWARD PASS - HIDDEN
def t_bck_hid(pi, beta, L, P):
    result = 0
    for l in range(1, L - 1):
        result += sequential.t_bh_delta(pi, beta, P[l], P[l + 1]) + t_b_updbias(beta, P[l]) + t_b_updweights(pi, beta, P[l - 1], P[l])
    return result


# Use what we can from sequential, but take different pi's and beta's into account
def predict(sample_parameters, machine_parameters, n_threads):
    L = int(sample_parameters[0]) + 2
    P = [int(sample_parameters[4])] + [int(sample_parameters[1])] * (L - 2) + [int(sample_parameters[5])]
    N = int(sample_parameters[2])
    S = int(sample_parameters[3])

    pi = machine_parameters[n_threads][0] * 1000000000
    beta = machine_parameters[n_threads][1] * 1000000000

    fwd_time = sequential.t_fwd(pi, beta, L, P)
    bck_out_time = t_bck_out(pi, beta, L, P)
    bck_hid_time = t_bck_hid(pi, beta, L, P)
    updates_time = sequential.t_updates(pi, beta, L, P)
    total_time = N * S * (fwd_time + bck_out_time + bck_hid_time + updates_time)

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]
