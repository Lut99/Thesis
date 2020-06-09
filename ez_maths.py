# EZ MATHS.py
#   by Anonymous
#
# Created:
#   6/9/2020, 4:09:53 PM
# Last edited:
#   6/9/2020, 4:35:02 PM
# Auto updated?
#   Yes
#
# Description:
#   This file allows one to easily calculate some formulas
#

import argparse


# ROOFLINE MODELS 
def r_bo_delta(pi, beta):
    return min(pi, beta * (1/4))

def r_bo_updbias(pi, beta):
    return min(pi, beta * (1/16))

def r_bo_updweights(pi, beta):
    return min(pi, beta * (1/12))

# FLOP PREDICTIONS
def f_bo_delta(p):
    return 16 * p

def f_bo_updbias(p):
    return p

def f_bo_updweights(p_i, p_i1):
    return 2 * p_i * p_i1

# TIME COMPUTATIONS
def t_bo_delta(p, pi, beta):
    return r_bo_delta(pi, beta) / f_bo_delta(p)


def t_bo_updbias(p, pi, beta):
    return r_bo_updbias(pi, beta) / f_bo_updbias(p)


def t_bo_updweights(p_i, p_i1, pi, beta):
    return r_bo_updweights(pi, beta) / f_bo_updweights(p_i, p_i1)

# TOTAL ESTIMATION
def r_bkc_out(p_i, p_i1, pi, beta):
    total = t_bo_delta(p_i, pi, beta) + t_bo_updbias(p_i, pi, beta) + t_bo_updweights(p_i, p_i1, pi, beta)
    return (t_bo_delta(p_i, pi, beta) / total) * r_bo_delta(pi, beta) + \
            (t_bo_updbias(p_i, pi, beta) / total) * r_bo_updbias(pi, beta) + \
            (t_bo_updweights(p_i, p_i1, pi, beta) / total) * r_bo_updweights(pi, beta)


def main(nodes, pi, beta):
    layers = len(nodes)
    avg_gflops = []
    total_time = 0
    for l in range(1, layers):
        n_i = nodes[l - 1]
        n_i1 = nodes[l]

        # Compute the average GFLOP/s for this layer & total time
        time = t_bo_delta(n_i, beta, pi) + t_bo_updbias(n_i, beta, pi) + t_bo_updweights(n_i, n_i1, pi, beta)
        avg = r_bkc_out(n_i, n_i1, pi, beta)

        # Add to the avg list & increment time
        avg_gflops.append((time, avg))
        total_time += time

    # Now compute the weighted average
    result = 0
    for weight, gflops in avg_gflops:
        result += (weight / total_time) * gflops
    
    print(f"The predicted, average peak performance for the backward pass - output layer is: {result} GFLOP/s")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pi", type=float, required=True, help="The peak performance (in GFLOP/s) for the target machine")
    parser.add_argument("-b", "--beta", type=float, required=True, help="The peak memory bandwidth (in GB/s) for the target machine")
    parser.add_argument("-n", "--nodes", nargs="+", type=int, required=True, help="Number of nodes for every layer (including input & output)")

    args = parser.parse_args()

    exit(main(args.nodes, args.pi, args.beta))
