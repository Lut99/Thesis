"""
    SEQUENTIAL.py
        by Tim Müller
    
    This file implements the sequential analytical model.
"""


# Data obtained from http://nicolas.limare.net/pro/notes/2014/12/16_math_speed/
F_DIV = 6
F_EXP = 12



# FORWARD PASS
def t_fwd(pi, beta, L, P):
    f = f_fwd(L, P)
    b = b_fwd(L, P)

    if pi >= beta * (f / b):
        return f / pi
    else:
        return b / pi
    
def f_fwd(L, P):
    result = 0
    for l in range(1, L):
        result += P[l] * (2 * P[l - 1] + 2 + F_DIV + F_EXP)
    return result

def b_fwd(L, P):
    result = 0
    for l in range(1, L):
        result += P[l] * (16 + 16 * P[l - 1])
    return result



# BACKWARD PASS (COMMON)
def t_b_updbias(pi, beta, p):
    if pi >= beta * (1 / 16):
        return p / pi
    else:
        return (16 * p) / beta

def t_b_updweights(pi, beta, p, p1):
    if pi >= beta * (1 / 12):
        return (2 * p * p1) / pi
    else:
        return (24 * p * p1) / beta



# BACKWARD PASS - OUTPUT
def t_bck_out(pi, beta, L, P):
    return t_bo_delta(pi, beta, P[L - 1]) + t_b_updbias(pi, beta, P[L - 1]) + t_b_updweights(pi, beta, P[L - 2], P[L - 1])

def t_bo_delta(pi, beta, p):
    if pi >= beta * (1 / 16):
        return p / pi
    else:
        return (16 * p) / beta



# BACKWARD PASS - HIDDEN
def t_bck_hid(pi, beta, L, P):
    result = 0
    for l in range(1, L - 1):
        result += t_bh_delta(pi, beta, P[l], P[l + 1]) + t_b_updbias(pi, beta, P[l]) + t_b_updweights(pi, beta, P[l - 1], P[l])
    return result

def t_bh_delta(pi, beta, p, p1):
    f = f_bh_delta(p, p1)
    b = b_bh_delta(p, p1)

    if pi >= beta * (f / b):
        return f / pi
    else:
        return b / beta

def f_bh_delta(p, p1):
    return p * (2 * p1 + 3)

def b_bh_delta(p, p1):
    return p * (16 * p1 + 16)



# UPDATES
def t_updates(pi, beta, L, P):
    result = 0
    for l in range(L - 1):
        result += t_u_bias(pi, beta, P[l]) + t_u_weight(pi, beta, P[l], P[l + 1])
    return result

def t_u_bias(pi, beta, p):
    if pi >= beta * (1 / 12):
        return (2 * p) / pi
    else:
        return (24 * p) / beta

def t_u_weight(pi, beta, p, p1):
    if pi >= beta * (1 / 12):
        return (2 * p * p1) / pi
    else:
        return (24 * p * p1) / beta



def predict(sample_parameters, machine_parameters):
    n_threads = int(sample_parameters.iloc[0])
    L = int(sample_parameters.iloc[1]) + 2
    P = [int(sample_parameters.iloc[5])] + [int(sample_parameters.iloc[2])] * (L - 2) + [int(sample_parameters.iloc[6])]
    N = int(sample_parameters.iloc[3])
    S = int(sample_parameters.iloc[4])

    pi = machine_parameters[0] * 1000000000
    n_cores = machine_parameters[1]
    n_avx = machine_parameters[2]
    beta = machine_parameters[3] * 1000000000

    # Adjust the pi based on the number of cores and number of elements handled simultaneously
    pi /= n_cores
    pi /= n_avx

    fwd_time = t_fwd(pi, beta, L, P)
    bck_out_time = t_bck_out(pi, beta, L, P)
    bck_hid_time = t_bck_hid(pi, beta, L, P)
    updates_time = t_updates(pi, beta, L, P)
    total_time = N * (S * (fwd_time + bck_out_time + bck_hid_time) + updates_time)

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]
