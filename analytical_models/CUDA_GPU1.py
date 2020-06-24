"""
    CUDA GPU1.py
        by Tim Müller
    
    This file implements the analytical model for the GPU-optimised
    implementation.
"""


import sys
import sequential


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
    pass



# Use what we can from sequential, but take different pi's and beta's into account
def predict(sample_parameters, machine_parameters, n_threads):
    L = int(sample_parameters[0]) + 2
    P = [int(sample_parameters[4])] + [int(sample_parameters[1])] * (L - 2) + [int(sample_parameters[5])]
    N = int(sample_parameters[2])
    S = int(sample_parameters[3])

    # Try to obtain all the benchmarks needed
    try:
        pi_G = machine_parameters["GPU"][0]
        beta_G = machine_parameters["GPU"][1]
        beta_C_G = machine_parameters["GPU"][2]
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

