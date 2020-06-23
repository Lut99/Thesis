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
    result = 0
    for l in range(1, L):
        result += P[l] * (t_f_sum(pi, beta, P[l - 1]) + t_f_act(pi, beta))
    return result

def t_f_sum(pi, beta, p):
    frac = (2 * p) / (8 + 16 * p)

    if pi <= beta * frac:
        return (2 * p) / pi
    else:
        return (8 + 16 * p) / beta
    
def t_f_act(pi, beta):
    f = 2 + F_DIV + F_EXP

    if pi <= beta * (f / 8):
        return f / pi
    else:
        return 8 / beta



# BACKWARD PASS - OUTPUT
def t_bck_out(pi, beta, L, P):
    return t_bo_delta(pi, beta, P[L - 1]) + t_b_updbias(pi, beta, P[L - 1]) + t_b_updweights(pi, beta, P[L - 2], P[L - 1])

def t_bo_delta(pi, beta, p):
    if pi <= beta * (1 / 6):
        return (4 * p) / pi
    else:
        return (24 * p) / beta

def t_b_updbias(pi, beta, p):
    if pi <= beta * (1 / 16):
        return p / pi
    else:
        return (16 * p) / beta

def t_b_updweights(pi, beta, p, p1):
    if pi <= beta * (1 / 12):
        return (2 * p * p1) / pi
    else:
        return (24 * p * p1) / beta


# BACKWARD PASS - HIDDEN
def t_bck_hid(pi, beta, L, P):
    result = 0
    for l in range(1, L - 1):
        result += t_bh_delta(pi, beta, P[l], P[l + 1]) + t_b_updbias(pi, beta, P[l]) + t_b_updweights(pi, beta, P[l - 1], P[l])
    return result

def t_bh_delta(pi, beta, p, p1):
    return p * (t_bh_sum(pi, beta, p1) + t_bh_comp(pi, beta))

def t_bh_sum(pi, beta, p):
    if pi <= beta * (1 / 8):
        return (2 * p) / pi
    else:
        return (16 * p) / beta

def t_bh_comp(pi, beta):
    if pi <= beta * (3 / 16):
        return 3 / pi
    else:
        return 16 / beta



# UPDATES
def t_updates(pi, beta, L, P):
    result = 0
    for l in range(L - 1):
        result += t_u_bias(pi, beta, P[l]) + t_u_weight(pi, beta, P[l], P[l + 1])
    return result

def t_u_bias(pi, beta, p):
    if pi <= beta * (1 / 12):
        return (2 * p) / pi
    else:
        return (24 * p) / beta

def t_u_weight(pi, beta, p, p1):
    if pi <= beta * (1 / 12):
        return (2 * p * p1) / pi
    else:
        return (24 * p * p1) / beta



def predict(sample_parameters, machine_parameters, n_threads):
    L = int(sample_parameters[0]) + 2
    P = [int(sample_parameters[4])] + [int(sample_parameters[1])] * (L - 2) + [int(sample_parameters[5])]
    N = int(sample_parameters[2])
    S = int(sample_parameters[3])

    pi = machine_parameters[n_threads][0] * 1000000000
    beta = machine_parameters[n_threads][1] * 1000000000

    fwd_time = t_fwd(pi, beta, L, P)
    bck_out_time = t_bck_out(pi, beta, L, P)
    bck_hid_time = t_bck_hid(pi, beta, L, P)
    updates_time = t_updates(pi, beta, L, P)
    total_time = N * (S * (fwd_time + bck_out_time + bck_hid_time) + updates_time)

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]
