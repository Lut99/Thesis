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
DEFAULT_OUTPUT_FILE = "rankings.csv"

# Fill in the machine performance data please (as (preak_performance, n_cores, n_avx_elements, peak_mem_bandwidth)) The bandwidth is over a single core.
# NOTE: home_desktop & macbook gflops from https://www.intel.com/content/dam/support/us/en/documents/processors/APP-for-Intel-Core-Processors.pdf
# NOTE: home_desktop & macbook gbs obtained with LIKWID.
MACHINES = {
    "home_desktop": (460.8, 16, 4, 14936.06),
    # "DAS5": (0.1, 32, 4, 0.1),
    "macbook": (46.4, 4, 4, 11163.92)
}


def main(models_path, csv_path, output_path):
    print("\n### PREDICTION TOOL for ANALYTICAL MODELS ###\n")

    print(f"Tool configuration:")
    print(f" - Models folder     : {models_path}")
    print(f" - Benchmark data    : \"{csv_path}\"")
    print("")

    print("Loading benchmark results... ", end="")
    sys.stdout.flush()
    data = pd.read_csv(csv_path)
    # Check if we got any
    if len(data) == 0:
        print(f"\n\nERROR: CSV file does not contain any benchmarks.", file=sys.stderr)
        return -1
    print(f"Done (loaded {len(data)} benchmark(s))")

    print("Loading model implementations... ", end="")
    sys.stdout.flush()
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
    sys.stdout.flush()
    pairs = {}
    for machine_id in data["machine_id"].unique():
        # Each machine
        machine_data = data[data["machine_id"] == machine_id]
        pairs[machine_id] = {}
        for variation in machine_data["variation"].unique():
            # Each machine / implementation pair
            pairs[machine_id][variation] = machine_data[machine_data["variation"] == variation].iloc[:, 2:].reset_index(drop=True)
    print(f"Done (identified {sum([len(pairs[machine_id]) for machine_id in pairs])} pair(s))")

    print("Preparing predictions... ", end="")
    sys.stdout.flush()
    n_iterations = 5
    parameters = None
    length = None
    for machine_id in pairs:
        for variation in pairs[machine_id]:
            avg_data = pairs[machine_id][variation].iloc[:, :]
            for i in range(0, len(avg_data), n_iterations):
                for j in range(1, n_iterations):
                    if i + j >= len(data):
                        print(f"\nERROR: List of benchmarks is not nicely dividable by {n_iterations}.")
                        return -1
                    avg_data.iloc[i, 10:16] += avg_data.iloc[i + j, 10:16]
                avg_data.iloc[i, 10:16] /= n_iterations
            avg_data = avg_data.iloc[::5].reset_index(drop=True)
            pairs[machine_id][variation] = avg_data.drop(columns=["iteration"])
            if length is None: length = len(pairs[machine_id][variation])
            elif length != len(pairs[machine_id][variation]):
                print("\nERROR: Not all hardware / variation pairs have the same length.", file=sys.stderr)
                return -1
            # Also extract the parameter list
            if parameters is None: parameters = pairs[machine_id][variation].iloc[:, :7]
            elif not parameters.iloc[:, 1:].equals(pairs[machine_id][variation].iloc[:, 1:7]):
                print("\nERROR: Not all parameters are the same for all implementation / hardware pairs.", file=sys.stderr)
                return -1
            pairs[machine_id][variation] = pairs[machine_id][variation].iloc[:, 7:]
    print(f"Done\n")

    print("Opening output file... ", end="")
    try:
        output = open(output_path, "w")
    except OSError as e:
        print(f"\nERROR: Could not open file \"{output_path}\": {e.strerror}\n")
        return -1
    print("Done")
    print("Writing CSV headers... ", end="")
    output.write("ranking_id,ranking_position,n_threads,n_hidden_layers,nodes_per_layer,n_epochs,n_samples,sample_size,n_classes,predicted_machine,predicted_variation,predicted_runtime,benchmarked_machine,benchmarked_variation,benchmarked_runtime\n")
    print("Done")

    # Time to predict
    for i, row in parameters.iterrows():
        # Loop through the machines & pairs
        pred_runtimes = []
        empi_runtimes = []
        for machine_id in pairs:
            for variation in pairs[machine_id]:
                pred_runtimes.append([machine_id, variation] + [runtime for runtime in models[variation].predict(row, MACHINES[machine_id])])
                empi_runtimes.append([machine_id, variation] + list(pairs[machine_id][variation].iloc[i, 1:]))

        # Sort both the predicted and benchmarked runtimes
        pred_runtimes = sorted(pred_runtimes, key=lambda elem: elem[2])
        empi_runtimes = sorted(empi_runtimes, key=lambda elem: elem[2])

        # Print 'em
        print(f"Ranking for input ({','.join([str(parameters[p][i]) for p in parameters])}):")
        for j, pred_pair in enumerate(pred_runtimes):
            empi_pair = empi_runtimes[j]

            text = f"  {j + 1}) {pred_pair[0].upper()} on {pred_pair[1].upper()} ({pred_pair[2]}s)"
            text += " " * (55 - len(text)) + "VS "
            text += f"  {empi_pair[0].upper()} on {empi_pair[1].upper()} ({empi_pair[2]}s)"
            print(text)

            # Write 'em to a file
            output.write(f"{i},{j},{','.join([str(elem) for elem in row])},{pred_pair[0]},{pred_pair[1]},{pred_pair[2]},{empi_pair[0]},{empi_pair[1]},{empi_pair[2]}\n")
        print("")

    output.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE, required=False, help=f"Path the output file to which the rankings will be written. (Default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("-c", "--csv", default=DEFAULT_CSV, required=False, help=f"Path to the CSV that contains benchmarks. (Default: {DEFAULT_CSV})")
    parser.add_argument("-M", "--models", default=DEFAULT_MODEL_FOLDER, required=False, help=f"Path to the folder containing the model implementations. (Default: {DEFAULT_MODEL_FOLDER})")

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
    exit(main(args.models, args.csv, args.output))
