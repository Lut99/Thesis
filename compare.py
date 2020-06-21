# COMPARE.py
#   by Anonymous
#
# Created:
#   21/06/2020, 14:08:50
# Last edited:
#   21/06/2020, 14:08:50
# Auto updated?
#   Yes
#
# Description:
#   Simple script that compares the runtimes line-by-line
#

import sys
import os
import argparse

import pandas as pd
from collections import defaultdict


def main(name1, name2, file1, file2):
    print("\n*** COMPARE.py ***\n")

    print("Configuration:")
    print(f" - File 1 : {file1} ({name1})")
    print(f" - File 2 : {file2} ({name2})")
    print("")

    print("Loading files... ", end=""); sys.stdout.flush()
    data_v1 = pd.read_csv(file1)
    data_v2 = pd.read_csv(file2)
    print(f"Done (loaded {len(data_v1) + len(data_v2)} benchmarks)")

    # Let's average and match them
    print("Preparing data... ", end=""); sys.stdout.flush()
    parameters = defaultdict(lambda: defaultdict(lambda: {"data": 0, "n": 0}))
    for ver, data in [(name1, data_v1), (name2, data_v2)]:
        for i, row in data.iterrows():
            # Get the parameter set as tuple
            parameter_set = tuple(row.iloc[3:10])
            parameters[parameter_set][ver]["data"] += data["total_runtime"].iloc[i]
            parameters[parameter_set][ver]["n"] += 1
    # Let's get rid of all that defaultdicts while averaging
    parameters = {params: {ver: parameters[params][ver]["data"] / parameters[params][ver]["n"] for ver in parameters[params]} for params in parameters}
    print("Done\n")

    # Compare them for all sets
    print(f"Comparing '{name1}' VS '{name2}'...")
    cases = 0
    total = 0
    for params in parameters:
        runtime1 = parameters[params][name1]
        runtime2 = parameters[params][name2]

        cases += 1 if runtime1 >= runtime2 else 0
        if runtime1 < runtime2:
            print(f" > Difference detected @ {params}: {runtime1}s VS {runtime2}s")
        total += 1
    
    print(f"RESULT: In {cases} out of {total} cases, {name1} was slower or the same as {name2}")

    print("\nDone.\n")

    return 0


if __name__ == "__main__":
    # Load the files
    parser = argparse.ArgumentParser()
    parser.add_argument("-1", "--file_1", required=False, default="benchmark_results/DAS5/das5_omp2.csv", help="The first file to compare (probably the slowest one)")
    parser.add_argument("-2", "--file_2", required=False, default="benchmark_results/DAS5/das5_omp7.csv", help="The second file to compare (probably the fastest one)")

    args = parser.parse_args()

    # Test if the files exist
    if not os.path.exists(args.file_1):
        print(f"ERROR: File '{args.file_1}' does not exist", file=sys.stderr)
        exit(-1)
    if not os.path.isfile(args.file_1):
        print(f"ERROR: File '{args.file_1}' is not a file", file=sys.stderr)
        exit(-1)
    
    if not os.path.exists(args.file_2):
        print(f"ERROR: File '{args.file_2}' does not exist", file=sys.stderr)
        exit(-1)
    if not os.path.isfile(args.file_2):
        print(f"ERROR: File '{args.file_2}' is not a file", file=sys.stderr)
        exit(-1)
    
    # Extract a name from the file
    name_1 = '.'.join(args.file_1.split("/")[-1].split(".")[:-1])
    name_2 = '.'.join(args.file_2.split("/")[-1].split(".")[:-1])

    exit(main(name_1, name_2, args.file_1, args.file_2))
