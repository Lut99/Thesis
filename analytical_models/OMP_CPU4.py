"""
    OMP CPU4.py
        by Tim Müller
    
    This file implements the analytical model for the fourth variation of the
    OpenMP CPU-optimised implementations.
"""


import sequential


# FORWARD PASS
def t_fwd(pi, beta, pi_simd, beta_simd, L, P):
    result = 0
    for l in range(1, L):
        result += P[l] * (sequential.t_f_sum(pi_simd, beta_simd, P[l - 1]) + sequential.t_f_act(pi, beta))
    return result



# BACKWARD PASS - HIDDEN
def t_bck_hid(pi, beta, pi_simd, beta_simd, L, P):
    result = 0
    for l in range(1, L - 1):
        result += t_bh_delta(pi, beta, pi_simd, beta_simd, P[l], P[l + 1]) + \
                  sequential.t_b_updbias(pi_simd, beta_simd, P[l]) + \
                  sequential.t_b_updweights(pi_simd, beta_simd, P[l - 1], P[l])
    return result

def t_bh_delta(pi, beta, pi_simd, beta_simd, p, p1):
    return p * (sequential.t_bh_sum(pi_simd, beta_simd, p1) + sequential.t_bh_comp(pi, beta))



# Use what we can from sequential, but take different pi's and beta's into account
def predict(sample_parameters, machine_parameters):
    n_threads = int(sample_parameters[0])
    L = int(sample_parameters[1]) + 2
    P = [int(sample_parameters[5])] + [int(sample_parameters[2])] * (L - 2) + [int(sample_parameters[6])]
    N = int(sample_parameters[3])
    S = int(sample_parameters[4])

    pi = machine_parameters[n_threads][0] * 1000000000
    beta = machine_parameters[n_threads][1] * 1000000000
    pi_simd = machine_parameters[n_threads][2] * 1000000000
    beta_simd = machine_parameters[n_threads][3] * 1000000000

    fwd_time = t_fwd(pi, beta, pi_simd, beta_simd, L, P)
    bck_out_time = sequential.t_bck_out(pi_simd, beta_simd, L, P)
    bck_hid_time = t_bck_hid(pi, beta, pi_simd, beta_simd, L, P)
    updates_time = sequential.t_updates(pi_simd, beta_simd, L, P)
    total_time = N * (S * (fwd_time + bck_out_time + bck_hid_time) + updates_time)

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]
