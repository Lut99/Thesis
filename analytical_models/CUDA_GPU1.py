"""
    CUDA GPU1.py
        by Tim Müller
    
    This file implements the analytical model for the GPU-optimised
    implementation.
"""


import sys
import sequential
import math


# INITIALIZATION
def t_init(beta_C_G, S, L, P):
    return t_nn_params(beta_C_G, L, P) + t_inputs(beta_C_G, S, P[0]) + t_expected(beta_C_G, S, P[L - 1])

def t_nn_params(beta_C_G, L, P):
    result = 0
    for l in range(1, L):
        result += (8 * P[l] * (1 + P[l - 1])) / beta_C_G
    return result

def t_inputs(beta_C_G, S, p):
    return (8 * S * p) / beta_C_G

def t_expected(beta_C_G, S, p):
    return (8 * S * p) / beta_C_G



# BACKWARD PASS - OUTPUT
def t_bck_out(pi_G, beta_G, L, P):
    return sequential.t_bo_delta(pi_G, beta_G, P[L - 1]) + t_b_updweight(pi_G, beta_G, P[L - 2], P[L - 1])

def t_b_updweight(pi_G, beta_G, p, p1):
    if pi_G <= beta_G * (1 / 8):
        return (p * p1) / pi_G
    else:
        return (16 * p * p1) / pi_G

# BACKWARD PASS - HIDDEN
def t_bck_hid(pi_G, beta_G, L, P):
    result = 0
    for l in range(1, L - 1):
        result += sequential.t_bh_delta(pi_G, beta_G, P[l], P[l + 1]) + t_b_updweight(pi_G, beta_G, P[l - 1], P[l])
    return result

# UPDATES
def t_upd(pi_G, beta_G, S, L, P):
    result = 0
    for l in range(1, L):
        result += t_u_bias(pi_G, beta_G, S, P[l]) + t_u_weight(pi_G, beta_G, S, P[l - 1], P[l])
    return result

def t_u_bias(pi_G, beta_G, s, p):
    if s == 1:
        return p * t_update(pi_G, beta_G)
    else:
        return s * p * t_reduce(pi_G, beta_G) + t_u_bias(pi_G, beta_G, math.ceil(s / 2), p)

def t_u_weight(pi_G, beta_G, s, p, p1):
    if s == 1:
        return p * p1 * t_update(pi_G, beta_G)
    else:
        return s * p * p1 * t_reduce(pi_G, beta_G) + t_u_weight(pi_G, beta_G, math.ceil(s / 2), p, p1)

def t_reduce(pi_G, beta_G):
    if pi_G <= beta_G * (1 / 8):
        return 2 / pi_G
    else:
        return 16 / beta_G

def t_update(pi_G, beta_G):
    if pi_G <= beta_G * (1 / 16):
        return 1 / pi_G
    else:
        return 16 / beta_G



# DOWNLOAD
def t_download(beta_C_G, L, P):
    result = 0
    for l in range(1, L):
        result += (8 * P[l] * (1 + P[l - 1])) / beta_C_G
    return result



# Use what we can from sequential, but take different pi's and beta's into account
def predict(sample_parameters, machine_parameters, n_threads):
    L = int(sample_parameters[0]) + 2
    P = [int(sample_parameters[4])] + [int(sample_parameters[1])] * (L - 2) + [int(sample_parameters[5])]
    N = int(sample_parameters[2])
    S = int(sample_parameters[3])

    # Try to obtain all the benchmarks needed
    try:
        pi_G = machine_parameters["GPU"][0] * 1000000000
        beta_G = machine_parameters["GPU"][1] * 1000000000
        beta_C_G = machine_parameters["GPU"][2] * 1000000000
    except KeyError:
        print("ERROR: Executing CUDA model for machine which has no GPU performance statistics defined.", file=sys.stderr)
        exit(-1)

    init_time = t_init(beta_C_G, S, L ,P)
    fwd_time = sequential.t_fwd(pi_G, beta_G, L, P)
    bck_out_time = t_bck_out(pi_G, beta_G, L, P)
    bck_hid_time = t_bck_hid(pi_G, beta_G, L, P)
    updates_time = t_upd(pi_G, beta_G, S, L, P)
    download_time = t_download(beta_C_G, L ,P)
    total_time = init_time + N * (S * (fwd_time + bck_out_time + bck_hid_time) + updates_time) + download_time

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]
