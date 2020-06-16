"""
    PREDICT COMPARE.py
        by Tim Müller
    
    This script tries to predict the optimal running time for a given
    application using different analytical models. For the input, it uses those
    as found in a benchmarking CSV, and shows us the difference between the
    predicted and actual runtimes.
"""

import argparse
import pandas as pd
import os
import sys


DEFAULT_CSV = "benchmark_results.csv"
DEFAULT_MODEL_FOLDER = "analytical_models/"

# Fill in the machine performance data please (as (GFLOP/s, GB/s))
MACHINES = {
    "home_desktop": (0, 0),
    "DAS5": (0, 0),
    "macTim": (0, 0)
}


def main(models_path, csv):
    print("\n### PREDICTION TOOL for ANALYTICAL MODELS ###\n")

    print(f"Tool configuration:")
    print(f" - Models folder     : {models_path}")
    print(f" - Benchmark data    : \"{csv}\"")
    print("")

    print("Loading benchmark results... ", end="")
    sys.stdout.flush()
    data = pd.read_csv(csv)
    # Check if we got any
    if len(data) == 0:
        print(f"\n\nERROR: CSV file does not contain any benchmarks.", file=sys.stderr)
        return -1
    print(f"Done (loaded {len(data)} benchmark(s))")

    print("Loading model implementations... ", end="")
    sys.path.append(models_path)
    models = {}
    for variation in data["variation"].unique():
        try:
            models[variation] = __import__(variation, fromlist=[variation])
        except ModuleNotFoundError:
            print(f"\n\nERROR: Could not load file '{variation}.py' from '{models_path}'.", file=sys.stderr)
            return -1
    print(f"Done (loaded {len(models)} models)")    

    print("Splitting into hardware / implementation pairs... ", end="")
    pairs = {}
    for machine_id in data["machine_id"].unique():
        # Each machine
        machine_data = data[data["machine_id"] == machine_id]
        pairs[machine_id] = {}
        for variation in machine_data["variation"].unique():
            # Each machine / implementation pair
            pairs[machine_id][variation] = machine_data[machine_data["variation"] == variation].iloc[:, 2:].reset_index(drop=True)
    print(f"Done (identified {sum([len(pairs[machine_id]) for machine_id in pairs])} pair(s))")

    print("Averaging predictions... ", end="")
    n_iterations = 5
    length = None
    for machine_id in pairs:
        for variation in pairs[machine_id]:
            avg_data = pairs[machine_id][variation].iloc[:, :]
            for i in range(0, len(avg_data), n_iterations):
                for j in range(1, n_iterations):
                    if i + j >= len(data):
                        print(f"\nWARNING: List of benchmarks is not nicely dividable by {n_iterations} (last average is probably incorrect).")
                        break
                    avg_data.iloc[i, 10:16] += avg_data.iloc[i + j, 10:16]
                avg_data.iloc[i, 10:16] /= n_iterations
            avg_data = avg_data.iloc[::5].reset_index(drop=True)
            pairs[machine_id][variation] = avg_data.drop(columns=["iteration"])
            if length is None: length = len(pairs[machine_id][variation])
            elif length != len(pairs[machine_id][variation]):
                print("ERROR: Not all hardware / variation pairs have the same length.", file=sys.stderr)
    print(f"Done\n")

    # Time to predict
    # TBD

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--csv", default=DEFAULT_CSV, required=False, help=f"Path to the CSV that contains benchmarks. (Default: {DEFAULT_CSV})")
    parser.add_argument("-M", "--models", default=DEFAULT_MODEL_FOLDER, required=False, help=f"Path to the folder containing the model implementations. (DEFAULT: {DEFAULT_MODEL_FOLDER}")

    args = parser.parse_args()
    
    # Check the validity of the CSV file
    if not os.path.exists(args.csv):
        print(f"ERROR: File '{args.csv}' does not exist.", file=sys.stderr)
        exit(-1)
    if not os.path.isfile(args.csv):
        print(f"ERROR: File '{args.csv}' is not a file.", file=sys.stderr)
        exit(-1)
    
    # Check if the models folder exists
    if not os.path.exists(args.models):
        print(f"ERROR: Folder '{args.models}' does not exist.", file=sys.stderr)
        exit(-1)
    if not os.path.isdir(args.models):
        print(f"ERROR: Folder '{args.models}' is not a folder.", file=sys.stderr)
        exit(-1)

    # So far so good, let's run main
    exit(main(args.models, args.csv))
