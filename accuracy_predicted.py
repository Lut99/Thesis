"""
    ACCURACY PREDICTED.py
        by Tim Müller

    A script that is used to predict the accuracy of a predicted ranking and
    the expected ranking. Note that it reads the file outputted by
    predict_compare.py.
"""

import os
import sys
import argparse
import pandas as pd
from collections import defaultdict


DEFAULT_RANKING_LOCATION = "rankings.csv"
DEFAULT_PODIUM_SIZE = 3


def main(rankings_path, ranking_id, podium_size):
    print("\n*** ACCURACY computation for RANKINGS ***\n")

    print("Configuration:")
    print(f" - Rankings location : '{rankings_path}'")
    print(f" - ID to check       : {ranking_id}")
    print(f" - Size of podium    : {podium_size}")
    print("")

    print("Loading rankings... ", end=""); sys.stdout.flush()
    rankings = pd.read_csv(rankings_path)
    print("Done")

    print("Fetching ranking... ", end=""); sys.stdout.flush()
    # Fetch all rankings by ID, then put 'em in a dict
    params = None
    ranking = {"predicted_rowwise": {}, "benchmarked_rowwise": {}, "predicted_samplewise": {}, "benchmarked_samplewise": {}}
    for i, row in rankings[rankings["ranking_id"] == ranking_id].iterrows():
        if params is None: params = tuple(row.iloc[2:8])
        elif tuple(row.iloc[2:8]) != params:
            print(f"ERROR: Parameters for row {i} are unexpected.", file=sys.stderr)
            return -1
        ranking["predicted_rowwise"][row["ranking_position"]] = tuple(row.iloc[8:12])
        ranking["benchmarked_rowwise"][row["ranking_position"]] = tuple(row.iloc[12:16])
        ranking["benchmarked_samplewise"][tuple(row.iloc[12:15])] = row.iloc[15]
    print("Done\n")

    print("Computing accuracy... ", end=""); sys.stdout.flush()
    # This is where the magic happens!
    # METRIC 1: Absolute rank position: if they are correct they are correct, and if they are not they are not
    accuracy = 0
    total = 0
    for rank in ranking["predicted_rowwise"]:
        if ranking["predicted_rowwise"][rank][:3] == ranking["benchmarked_rowwise"][rank][:3]:
            accuracy += 1
        total += 1
    # Make it into an accuracy
    accuracy = accuracy / total * 100
    print("Done")

    print("Computing performance loss... ", end=""); sys.stdout.flush()
    # This is where the magic happens!
    # METRIC 2: Performance loss: we given the model points off for the benchmarked time of predicted best VS actual best
    performance_loss = defaultdict(int)
    total = 0
    for rank in range(min(len(ranking["predicted_rowwise"]), podium_size)):
        # Get the position of the actual best rank
        predicted_best = ranking["benchmarked_samplewise"][ranking["predicted_rowwise"][rank][:3]]
        actual_best = ranking["benchmarked_rowwise"][rank][3]
        performance_loss[rank] += abs(predicted_best - actual_best)
        performance_loss["total"] += abs(predicted_best - actual_best)
    print("Done\n")

    line_size = 60
    print(f"Ranking {ranking_id}:")
    print("  Predicted:" + " " * (line_size - 6) + "Benchmarked:")
    for i in ranking["predicted_rowwise"]:
        pm, pv, pt, pr = ranking["predicted_rowwise"][i]
        bm, bv, bt, br = ranking["benchmarked_rowwise"][i]

        text = f"   {i + 1}) {pm.upper()} ({pt} thread{'' if pt == 1 else 's'}) with {pv.upper()} ({pr:.4f}s)"
        text += " " * (line_size - len(text)) + " VS "
        text += f"   {i + 1}) {bm.upper()} ({bt} thread{'' if bt == 1 else 's'}) with {bv.upper()} ({br:.4f}s)"
        print(text)

    print(f"Accuracy               : {accuracy:.2f}%")
    print(f"Total performance loss : {performance_loss['total']:.6f}s")
    print(" - Per ranking:")
    for rank, runtime in {rank: performance_loss[rank] for rank in performance_loss if rank != 'total'}.items():
        print(f"    - Rank {rank + 1:02d}          : {runtime:.6f}s")

    print("\nDone.\n")

    return 0


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--rankings", required=False, default=DEFAULT_RANKING_LOCATION, help="The location of the file containing the rankings.")
    parser.add_argument("-i", "--id", required=True, type=int, help="The ID of the ranking that we want to compare")
    parser.add_argument("-p", "--podium_size", required=False, type=int, default=DEFAULT_PODIUM_SIZE, help=f"Top N rankings to check for the performance loss metric. (DEFAULT: {DEFAULT_PODIUM_SIZE})")

    args = parser.parse_args()

    # Check if rankings exist
    if not os.path.exists(args.rankings):
        print(f"ERROR: File '{args.rankings}' does not exist.", file=sys.stderr)
        exit(-1)
    if not os.path.isfile(args.rankings):
        print(f"ERROR: File '{args.rankings}' is not a file.", file=sys.stderr)
        exit(-1)

    exit(main(args.rankings, args.id, args.podium_size))
