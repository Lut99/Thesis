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

import pandas as pd
from collections import defaultdict


if __name__ == "__main__":
    data_omp2 = pd.read_csv("benchmark_results/DAS5/das5_omp2.csv")
    data_omp7 = pd.read_csv("benchmark_results/DAS5/das5_omp7.csv")

    # Let's average and match them
    parameters = defaultdict(lambda: defaultdict(int))
    for ver, data in [("ver1", data_omp2), ("ver2", data_omp7)]:
        n_iters = 0
        for i, row in data.iterrows():
            # Get the parameter set as tuple
            parameter_set = tuple(row.iloc[3:10])
            parameters[parameter_set][ver] += data["total_runtime"].iloc[i]
            if data["iteration"].iloc[i] > n_iters:
                n_iters = data["iteration"].iloc[i]
        # Don't forget to normalize to average
        parameters[parameter_set][ver] /= n_iters
    # Let's get rid of all that defaultdicts
    parameters = {params: {ver: parameters[params][ver] for ver in parameters[params]} for params in parameters}

    # Compare them for all sets
    cases = 0
    total = 0
    for params in parameters:
        runtime_slow = parameters[params]["ver1"]
        runtime_fast = parameters[params]["ver2"]

        cases += 1 if runtime_slow >= runtime_fast else 0
        if runtime_slow < runtime_fast:
            print(f"Difference detected @ {params}: {runtime_slow}s VS {runtime_fast}s")
        total += 1
    
    print(f"In {cases} out of {total} cases, OpenMP variation 7 was faster or the same as OpenMP variation 2")
