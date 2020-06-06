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
import subprocess
from collections import defaultdict


# Constants
DEFAULT_OUTPUTPATH = "benchmark_results.csv"
DEFAULT_CODEDIR = "src/lib/NeuralNetwork"
DEFAULT_VARIATIONS = ["*"]
DEFAULT_THREADS = [1, 2, 4, 8, 16, 32]
DEFAULT_ITERATIONS = 10
HEADERS = ["num_threads"]


def is_float(n):
    # Returns if n is parsable to a float
    try:
        float(n)
        return True
    except ValueError:
        return False


def run(var_ID, params, das_reservation):
    cmd = ["bin/digits.out", "./digits.csv"] + [str(params[header]) for header in HEADERS if header in params]
    if das_reservation is not None:
        cmd = ["prun", "-np", "1", "-reserve", f"{das_reservation}"] + cmd

    # Runs a single testcase
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if it was succesful
    if result.returncode != 0:
        print(f"\nERROR: Failed to run '{' '.join(cmd)}' (return status {result.returncode}):\n{result.stderr.decode('utf-8')}.", file=sys.stderr)
        exit(-1)
    
    # Fetch the result and try to parse the Time taken from it
    out = result.stdout.decode('utf-8')
    data = out.split(",")
    if len(data) != 2 or not is_float(data[0]) or not is_float(data[1]):
        print(f"\nERROR: Failed to retrieve performance data from '{' '.join(cmd)}': expected two numbers separated by comma, got: \"{out}\".", file=sys.stderr)
        exit(-1)

    return float(data[0]), float(data[1])


def compile(var_ID):
    # Clean existing bin/digits.out file to force linking
    res = subprocess.run(["rm", "-f", "bin/digits.out"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"\nERROR: Failed to run 'rm -f bin/digits.out' (return status {res.returncode}):\n{res.stderr.decode('utf-8')}", file=sys.stderr)
        exit(-1)

    # Compile new variation
    res = subprocess.run(["make", "digits", f"VARIATION={var_ID}", "BENCHMARK=1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"\nERROR: Failed to run 'make digits VARIATION={var_ID}' (return status {res.returncode}):\n{res.stderr.decode('utf-8')}", file=sys.stderr)
        exit(-1)


def run_benchmark(outputfile, var_ID, iterations, params, das_reservation,):
    avg_runtime = 0
    for i in range(iterations):
        if len(params) > 0:
            print("   ", end="")
        print(f"         Iter ({i + 1}/{iterations})...", end="")
        sys.stdout.flush()

        # Run it
        runtime, cputime = run(var_ID, params, das_reservation)
        print(f" {runtime}s")

        # Write the info about this run and the result to the file
        print(f"{var_ID},{i}", file=outputfile, end="")
        for header in HEADERS:
            if header in params:
                print(f",{params[header]}", file=outputfile, end="")
            else:
                # For headers that are not present in the current parameters, just print a '-'
                print(f",-", file=outputfile, end="")
        print(f",{runtime},{cputime}", file=outputfile)

        avg_runtime += runtime
    if len(params) > 0:
        print("   ", end="")
    print(f"       > Average: {avg_runtime / iterations} seconds")


def main(outputpath, variations, threads, iterations, das_reservation):
    print("\n### BENCHMARK TOOL for NEURALNETWORK.c ###\n")

    print(f"Configuration:")
    print(f" - Output file           : \"{outputpath}\"")
    print(f" - Benchmarking versions : {variations}")
    print(f" - Threads to test       : {threads}")
    print(f" - Iterations per test   : {iterations}")
    print(f" - DAS5-mode             : {das_reservation is not None}")
    if das_reservation is not None:
        print(f"    - DAS5 reservation   : {das_reservation}")
    print("")

    print("Cleaning existing binaries...", end="")
    res = subprocess.run(["make", "clean"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"\nERROR: Failed to run 'clean make' (return status {res.returncode}):\n{res.stderr.decode('utf-8')}", file=sys.stderr)
        return -1
    print(" Done\n")

    print("Acquiring output file...", end="")
    try:
        output = open(outputpath, "w")
    except OSError as e:
        print(f" FAIL\nERROR: Could not open file \"{outputpath}\": {e.strerror}\n")
        return -1
    print(" Done\n")

    print("Writing headers...", end="")
    print("variation,iteration," + ",".join(HEADERS) + ",runtime,cputime", file=output)
    print(" Done\n")

    print("Splitting variations by functionality...")
    seqs = []
    cpus = []
    gpus = []
    unks = []
    for variation in variations:
        is_seq = "seq" in variation.lower()
        is_cpu = "cpu" in variation.lower()
        is_gpu = "gpu" in variation.lower()

        if is_seq:
            seqs.append(variation)
        if is_cpu:
            cpus.append(variation)
        if is_gpu:
            gpus.append(variation)
        if not is_seq and not is_cpu and not is_gpu:
            # Mark as unknown
            unks.append(variation)
        
    if len(seqs) > 0:
        print(f" > Found {len(seqs)} variation(s) that are sequentially implemented")
    if len(cpus) > 0:
        print(f" > Found {len(cpus)} variation(s) that are optimised for the CPU")
    if len(gpus) > 0:
        print(f" > Found {len(gpus)} variation(s) that are optimised for the GPU")
    if (len(unks) > 0):
        print(f"\nWARNING: Found {len(unks)} variation(s) that cannot be identified: {unks}\n")
    print("Done\n")

    # First, benchmark the sequential ones
    if len(seqs) > 0:
        print("Performing benchmarks for sequential implementations...")
        for seq in seqs:
            print(f" > Variation: {seq}")

            # Compile first
            print("      Compiling...", end="")
            sys.stdout.flush()
            compile(seq)
            print(" Done")

            # Run for the different parameters
            print("      Running...")
            run_benchmark(output, seq, iterations, {}, das_reservation)
            print("      Done")
        print("Done\n")

    # Next up: CPU benchmarks
    if len(cpus) > 0:
        print("Performing benchmarks for CPU implementations...")
        for cpu in cpus:
            print(f" > Variation: {cpu}")

            # Compile first
            print("      Compiling...", end="")
            sys.stdout.flush()
            compile(cpu)
            print(" Done")

            # Run for the different parameters
            print("      Running...")
            for n_threads in threads:
                print(f"       > Params: n_threads={n_threads}")
                run_benchmark(output, cpu, iterations, {"num_threads": n_threads}, das_reservation)
            print("      Done")
        print("Done\n")

    # Finally, GPU benchmarks
    if len(gpus) > 0:
        print("Performing benchmarks for GPU implementations")
        print("Done\n")

    # Close output file
    output.close()

    print("Done.\n")

    return 0


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Benchmark for the Feedforward Neural Network.')
    parser.add_argument("-r", "--reservation", required=False, default=None, type=int, help="If given, uses the DAS 'prun' command to run a benchmark on the remote node with the given reservation number.")
    parser.add_argument("-o", "--output", required=False, default=DEFAULT_OUTPUTPATH, help=f"Path to file that contains the data. Note that any existing files will be overwritten. Default: \"{DEFAULT_OUTPUTPATH}\"")
    parser.add_argument("-d", "--directory", required=False, default=DEFAULT_CODEDIR, help=f"The path to the directory with all the NeuralNetwork implementations. Default: \"{DEFAULT_CODEDIR}\"")
    parser.add_argument("-v", "--variations", required=False, nargs='+', default=DEFAULT_VARIATIONS, help=f"The different code variations that will be benchmarked. The version given should equal the filename of the target file, excluding 'NeuralNetwork_' and '.c'. If set to \"*\", all files in the Optimisation directory are benchmarked. Default: {DEFAULT_VARIATIONS}")
    parser.add_argument("-t", "--threads", required=False, nargs='+', default=DEFAULT_THREADS, type=int, help=f"Specify the values for which to try different number of threads. Default: {DEFAULT_THREADS}")
    parser.add_argument("-i", "--iterations", required=False, default=DEFAULT_ITERATIONS, type=int, help=f"The number of iterations each test case will be run. Default: {DEFAULT_ITERATIONS}")

    args = parser.parse_args()


    # Check if the prun library exists if we do so
    if args.reservation is not None:
        try:
            subprocess.run(["prun", "--help"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(f"ERROR: 'prun' module not loaded or not on DAS5.", file=sys.stderr)
            exit(-1)


    # Check if the outputpath is not a dir
    if os.path.isdir(args.output):
        print(f"ERROR: Given output file \"{args.output}\" is a directory.", file=sys.stderr)
        exit(-1)


    # Check if the directory exists
    if not os.path.exists(args.directory):
        print(f"ERROR: Given directory \"{args.directory}\" does not exist.", file=sys.stderr)
        exit(-1)
    elif not os.path.isdir(args.directory):
        print(f"ERROR: Given directory \"{args.directory}\" is not a directory.", file=sys.stderr)
        exit(-1)


    # Find all files if told to do so. Otherwise, check if files exist and sort them.
    if len(args.variations) == 1 and args.variations[0] == "*":
        args.variations = []
        for path in os.listdir(args.directory):
            # Make sure the file start with NeuralNetwork_ and ends with .c
            if len(path) < 16 or path[:14] != "NeuralNetwork_" or path[-2:] != ".c":
                print(f"WARNING: Ignoring file \"{path}\".")
            
            # Extract the variation ID and add it the list
            args.variations.append(path[14:-2])
        
        # Sort the list to achieve somewhat constant and logical ordering
        args.variations = sorted(args.variations)
    else:
        # First, check if the files exist
        for var_ID in args.variations:
            path = os.path.join(args.directory, "NeuralNetwork_" + var_ID + ".c")
            if not os.path.exists(path):
                print(f"ERROR: File \"{path}\" does not exist.", file=sys.stderr)
                exit(-1)
            elif not os.path.isfile(path):
                print(f"ERROR: File \"{path}\" is not a file.", file=sys.stderr)
                exit(-1)
        
        # Order them
        args.variations = sorted(args.variations)


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
    exit(main(args.output, args.variations, args.threads, args.iterations, args.reservation))
