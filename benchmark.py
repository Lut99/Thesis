"""
    BENCHMARK.py
        by Tim Müller (11774606)
    
    Description:
    This file implements an extensive benchmark of the Feedforward Neural
    Network. This file runs the benchmark for a number of varying parameters,
    for each version, multiple times. Be warned that execution may take a
    while.
"""

import argparse
import os
import sys


DEFAULT_CODEDIR = "src/lib/NeuralNetwork"
DEFAULT_VERSIONS = ["*"]
DEFAULT_THREADS = [1, 2, 4, 8, 16, 32]
DEFAULT_ITERATIONS = 10


def run_iter(num_threads, ver_ID):
    # Runs a single testcase


def main(versions, threads, iterations):
    print("\n### BENCHMARK TOOL for NEURALNETWORK.c ###\n")

    print(f"Configuration:")
    print(f"  - Benchmarking versions : {versions}")
    print(f"  - Threads to test       : {threads}")
    print("")

    print("Starting benchmark...")
    for num_threads in threads:
        print("  > NEW config (n_threads={num_threads}):")
        for ver_ID in versions:
            print(f"      - Version '{ver_ID}'")
            avg_runtime = 0
            for i in range(iterations):
                print(f"        (Iter {i + 1}/{iterations})", end="\r")

                # Run it
                avg_runtime += run_iter(num_threads, ver_ID)

            print(f"        Result: {avg_runtime / iterations} seconds")

    return 0


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Benchmark for the Feedforward Neural Network.')
    parser.add_argument("-d", "--directory", required=False, default=DEFAULT_CODEDIR, help=f"The path to the directory with all the NeuralNetwork implementations. Default: \"{DEFAULT_CODEDIR}\"")
    parser.add_argument("-v", "--versions", required=False, nargs='+', default=DEFAULT_VERSIONS, help=f"The OpenMP versions that are benchmarked. The version given should equal the filename of the target file, excluding 'NeuralNetwork_' and '.c'. If set to \"*\", all files in the Optimisation directory are benchmarked. Default: {DEFAULT_VERSIONS}")
    parser.add_argument("-t", "--threads", required=False, nargs='+', default=DEFAULT_THREADS, type=int, help=f"Specify the values for which to try different number of threads. Default: {DEFAULT_THREADS}")
    parser.add_argument("-i", "--iterations", required=False, default=DEFAULT_ITERATIONS, type=int, help=f"The number of iterations each test case will be run. Default: {DEFAULT_ITERATIONS}")

    args = parser.parse_args()


    # Check if the directory exists
    if not os.path.exists(args.directory):
        print(f"ERROR: Given directory \"{args.directory}\" does not exist.", file=sys.stderr)
        exit(-1)
    elif not os.path.isdir(args.directory):
        print(f"ERROR: Given directory \"{args.directory}\" is not a directory.", file=sys.stderr)
        exit(-1)


    # Find all files if told to do so. Otherwise, check if files exist and sort them.
    if len(args.versions) == 1 and args.versions[0] == "*":
        args.versions = []
        for path in os.listdir(args.directory):
            # Make sure the file start with NeuralNetwork_ and ends with .c
            if len(path) < 16 or path[:14] != "NeuralNetwork_" or path[-2:] != ".c":
                print(f"WARNING: Ignoring file \"{path}\".")
            
            # Extract the variation ID and add it the list
            args.versions.append(path[14:-2])
        
        # Sort the list to achieve somewhat constant and logical ordering
        args.versions = sorted(args.versions)

        # Find sequential, and if found, put it first
        if "Sequential" in args.versions:
            args.versions.remove("Sequential")
            args.versions = ["Sequential"] + args.versions
    else:
        # First, check if the files exist
        for var_ID in args.versions:
            path = os.path.join(args.directory, "NeuralNetwork_" + var_ID + ".c")
            if not os.path.exists(path):
                print(f"ERROR: File \"{path}\" does not exist.", file=sys.stderr)
                exit(-1)
            elif not os.path.isfile(path):
                print(f"ERROR: File \"{path}\" is not a file.", file=sys.stderr)
                exit(-1)
        
        # Order them
        args.versions = sorted(args.versions)

        # Make sure sequential is the head
        if "Sequential" in args.versions:
            args.versions.remove("Sequential")
            args.versions = ["Sequential"] + args.versions


    # Check if the number of threads make sense
    for t in args.threads:
        if t < 1:
            print(f"ERROR: Number of threads can only be positive, not {t}.", file=sys.stderr)
            exit(-1)

    # Check if the number of iterations is legal
    if args.iterations < 1:
        print(f"ERROR: Number of iterations can only be positive, not {args.iterations}.", file=sys.stderr)
        exit(-1)


    # Now that's done, call main
    exit(main(args.versions, args.threads, args.iterations))
