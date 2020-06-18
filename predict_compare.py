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
import numpy as np
import os
import sys
from collections import defaultdict


DEFAULT_CSV = "benchmark_results.csv"
DEFAULT_MODEL_FOLDER = "analytical_models/"
DEFAULT_OUTPUT_FILE = "rankings.csv"

# Fill in the machine performance data please (as (preak_performance, n_cores, n_avx_elements, peak_mem_bandwidth)) The bandwidth is over a single core.
# NOTE: hall GFLOP/s from https://www.intel.com/content/dam/support/us/en/documents/processors/APP-for-Intel-Core-Processors.pdf
# NOTE: all GB/s obtained with LIKWID (copy or copy-avx, 1GB load).
# NOTE: First two are non-simd, second two are yes-simd
MACHINES = {
    "home_desktop": {
        1: (7.2, 14.93606, 28.8, 14.33228),
        2: (14.4, 15.86464, 57.6, 15.65240),
        4: (28.8, 16.50343, 115.2, 17.38215),
        8: (57.6, 16.94652, 230.4, 17.19413),
        16: (115.2, 17.09865, 460.8, 17.09826),
        32: (115.2, 17.01528, 460.8, 17.18359)
    },
    "DAS5_1numa": {
        1: (2.4, 11.53074, 9.6, 11.42862),
        2: (4.8, 12.49312, 19.2, 12.27604),
        4: (9.6, 22.43777, 38.4, 20.62895),
        8: (19.2, 30.47322, 76.8, 27.52148),
        16: (38.4, 31.12496, 153.6, 28.11800),
        32: (76.8, 31.10337, 307.2, 28.10550)
    },
    "DAS5": {
        1: (2.4, 11.53074, 9.6, 11.42862),
        2: (4.8, 22.61996, 19.2, 21.99305),
        4: (9.6, 24.99948, 38.4, 24.37255),
        8: (19.2, 45.96919, 76.8, 41.58752),
        16: (38.4, 60.74498, 153.6, 55.02765),
        32: (76.8, 62.18000, 307.2, 56.00466)
    },
    "macbook": {
        1: (5.8, 11.16392, 11.6, 11.26597),
        2: (11.6, 10.22710, 23.2, 10.38897),
        4: (23.2, 11.60071, 46.4, 11.72698),
        8: (23.2, 11.54814, 46.4, 11.68686),
        16: (23.2, 11.55648, 46.4, 11.67511),
        32: (23.2, 11.55118, 46.4, 11.57948)
    }
}


def main(models_path, csv_files, output_path):
    print("\n### PREDICTION TOOL for ANALYTICAL MODELS ###\n")

    print(f"Tool configuration:")
    print(f" - Models folder       : {models_path}")
    print(f" - Benchmark datafiles :")
    for csv in csv_files:
        print(f"    - '{csv}'")
    print("")

    print("Loading benchmark results... ", end="")
    sys.stdout.flush()
    data = pd.DataFrame()
    for csv in csv_files:
        csv_data = pd.read_csv(csv)
        data = pd.concat([data, csv_data], ignore_index=True)
    # Check if we got any
    if len(data) == 0:
        print(f"\n\nERROR: No benchmarks found in any of the given files.", file=sys.stderr)
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

    print("Reading benchmarks... ", end="")
    sys.stdout.flush()
    parameters = defaultdict(lambda: defaultdict(lambda: {"n": 0, "data": None}))
    for i, row in data.iterrows():
        # Get the parameter set as tuple
        hw_imp_pair = tuple(row.iloc[0:2])
        parameter_set = tuple(row.iloc[3:10])
        if parameters[parameter_set][hw_imp_pair]["n"] == 0:
            parameters[parameter_set][hw_imp_pair]["data"] = np.array(row.iloc[10:])
        else:
            parameters[parameter_set][hw_imp_pair]["data"] += np.array(row.iloc[10:])
        parameters[parameter_set][hw_imp_pair]["n"] += 1
    # Once done, average all things
    parameters = {
        params: {
            hw_imp_pair: tuple(parameters[params][hw_imp_pair]["data"] / parameters[params][hw_imp_pair]["n"]) for hw_imp_pair in parameters[params]
        } for params in parameters
    }
    print("Done\n")

    print("Opening output file... ", end="")
    try:
        output = open(output_path, "w")
    except OSError as e:
        print(f"\nERROR: Could not open file \"{output_path}\": {e.strerror}\n")
        return -1
    print("Done")
    print("Writing CSV headers... ", end="")
    output.write("ranking_id,ranking_position,n_threads,n_hidden_layers,nodes_per_layer,n_epochs,n_samples,sample_size,n_classes,predicted_machine,predicted_variation,predicted_runtime,benchmarked_machine,benchmarked_variation,benchmarked_runtime\n")
    print("Done\n")

    # Time to predict
    print("Predicting and comparing...")
    total = 0
    correct = 0
    hw_total = defaultdict(int)
    hw_incorrect = defaultdict(int)
    var_total = defaultdict(int)
    var_incorrect = defaultdict(int)
    time_offset = defaultdict(lambda: defaultdict(list))
    for params in parameters:
        config_set = parameters[params]
        predicted_ranking = []
        benchmark_ranking = []
        for machine_id, variation in config_set:
            # Predict the runtime for this ranking
            predicted_ranking.append([machine_id, variation] + [runtime for runtime in models[variation].predict(params, MACHINES[machine_id])])
            # Also add the runtimes for the benchmarking
            benchmark_ranking.append([machine_id, variation] + [runtime for runtime in config_set[(machine_id, variation)]])
        
        # Sort both rankings to actually rank them
        predicted_ranking = sorted(predicted_ranking, key=lambda elem: elem[2])
        benchmark_ranking = sorted(benchmark_ranking, key=lambda elem: elem[3])

        # Compare if the machines are the same
        matches = True
        for i in range(len(config_set)):
            machine_id = predicted_ranking[i][0]
            variation = predicted_ranking[i][1]

            # Update the total
            hw_total[machine_id] += 1
            var_total[variation] += 1

            # Compute and store the time offset
            offset = benchmark_ranking[i][3] / predicted_ranking[i][2]
            time_offset[machine_id][variation].append(offset)

            if machine_id != benchmark_ranking[i][0] or variation != benchmark_ranking[i][1]:
                matches = False
                # Add an incorrect mark to both the hardware and the incorrect
                hw_incorrect[machine_id] += 1
                var_incorrect[variation] += 1
            
            # Print the ranking
            output.write(f"{total},{i},{','.join([str(p) for p in params])},{machine_id},{variation},{predicted_ranking[i][2]},{benchmark_ranking[i][0]},{benchmark_ranking[i][1]},{benchmark_ranking[i][3]}\n")
                
        
        # If they aren't, let the user know
        if not matches:
            # Print that they differ
            print(f"\nResults for parameters {params} differ:")
            print("  Predicted:" + " " * 44 + "Benchmarked:")
            for i in range(len(config_set)):
                text = f"   {i + 1}) {predicted_ranking[i][0].upper()} with {predicted_ranking[i][1].upper()} ({predicted_ranking[i][2]:.4f}s)"
                text += " " * (50 - len(text)) + " VS "
                text += f"   {i + 1}) {benchmark_ranking[i][0].upper()} with {benchmark_ranking[i][1].upper()} ({benchmark_ranking[i][2]:.4f}s)"
                print(text)
            print("")

        # Update the accuracy score
        total += 1
        correct += 1 if matches else 0

    print("\n***STATS***\n")
    print("Ranking accuracy:")
    print(f" - Overall accuracy: {(correct / total * 100):.2f}%")
    print(" - Accuracy per machine:")
    for m in hw_total:
        print(f"    - {m} : {(hw_total[m] - hw_incorrect[m]) / hw_total[m] * 100:.2f}%")
    print(" - Accuracy per variation:")
    for v in var_total:
        print(f"    - {v} : {(var_total[v] - var_incorrect[v]) / var_total[v] * 100:.2f}%")

    print("\nRatio of prediction time to benchmark time per machine, per variation:")
    for m in time_offset:
        print(f" - {m}:")
        for v in time_offset[m]:
            offsets = time_offset[m][v]
            print(f"    - {v}:")

            # Compute the mean first
            mean = sum(offsets) / len(offsets)
            print(f"       - mean     = {mean:.2f}x")

            # Then, the variance
            var = sum([(offset - mean) ** 2 for offset in offsets]) / len(offsets)
            print(f"       - variance = {var:.2f}")

    print("\nDone.\n")

    output.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE, required=False, help=f"Path the output file to which the rankings will be written. (Default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("-c", "--csvs", nargs='+', required=True, help=f"Files to load the benchmarks from. Can be more than one, which will be pasted internally after one another.")
    parser.add_argument("-M", "--models", default=DEFAULT_MODEL_FOLDER, required=False, help=f"Path to the folder containing the model implementations. (Default: {DEFAULT_MODEL_FOLDER})")

    args = parser.parse_args()
    
    # Check the validity of the CSV files
    for csv in args.csvs:
        if not os.path.exists(csv):
            print(f"ERROR: File '{csv}' does not exist.", file=sys.stderr)
            exit(-1)
        if not os.path.isfile(csv):
            print(f"ERROR: File '{csv}' is not a file.", file=sys.stderr)
            exit(-1)
    
    # Check if the models folder exists
    if not os.path.exists(args.models):
        print(f"ERROR: Folder '{args.models}' does not exist.", file=sys.stderr)
        exit(-1)
    if not os.path.isdir(args.models):
        print(f"ERROR: Folder '{args.models}' is not a folder.", file=sys.stderr)
        exit(-1)

    # So far so good, let's run main
    exit(main(args.models, args.csvs, args.output))
