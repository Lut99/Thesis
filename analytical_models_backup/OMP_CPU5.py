"""
    OMP CPU5.py
        by Tim Müller
    
    This file implements the analytical model for the fourth variation of the
    OpenMP CPU-optimised implementations.
"""


import sequential


# FORWARD PASS
def t_fwd(m_params, n_threads, L, P):
    result = 0
    print(m_params)
    for l in range(1, L):
        result += P[l] * (sequential.t_f_sum(m_params[n_threads][2], m_params[n_threads][3], P[l - 1]) + sequential.t_f_act(m_params[n_threads][0], m_params[n_threads][1]))
    return result



# BACKWARD PASS - HIDDEN
def t_bck_hid(t_params, L, P):
    result = 0
    for l in range(1, L - 1):
        result += t_bh_delta(t_params, P[l], P[l + 1]) + \
                  sequential.t_b_updbias(t_params[2], t_params[3], P[l]) + \
                  sequential.t_b_updweights(t_params[2], t_params[3], P[l - 1], P[l])
    return result

def t_bh_delta(t_params, p, p1):
    return p * (sequential.t_bh_sum(t_params[2], t_params[3], p1) + sequential.t_bh_comp(t_params[0], t_params[1]))



# Use what we can from sequential, but take different pi's and beta's into account
def predict(sample_parameters, machine_parameters):
    n_threads = int(sample_parameters[0])
    L = int(sample_parameters[1]) + 2
    P = [int(sample_parameters[5])] + [int(sample_parameters[2])] * (L - 2) + [int(sample_parameters[6])]
    N = int(sample_parameters[3])
    S = int(sample_parameters[4])

    # Copy the list of machine parameters and normalize to flops and bytes, not GFLOPs and GBYTES
    m_params = {n_cores: [elem * 1000000000 for elem in machine_parameters[n_cores]] for n_cores in machine_parameters}

    fwd_time = t_fwd(m_params, n_threads, L, P)
    bck_out_time = sequential.t_bck_out(m_params[1][2], m_params[1][3], L, P)
    bck_hid_time = t_bck_hid(m_params[1], L, P)
    updates_time = sequential.t_updates(m_params[n_threads][2], m_params[n_threads][3], L, P)
    total_time = N * (S * (fwd_time + bck_out_time + bck_hid_time) + updates_time)

    return [total_time, fwd_time, bck_out_time, bck_hid_time, updates_time]
