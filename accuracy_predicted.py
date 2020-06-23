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


DEFAULT_RANKING_LOCATION = "rankings.csv"


def main(rankings_path, ranking_id):
    print("\n*** ACCURACY computation for RANKINGS ***\n")

    print("Configuration:")
    print(f" - Rankings location : '{rankings_path}'")
    print(f" - ID to check       : {ranking_id}")
    print("")

    print("Loading rankings... ", end=""); sys.stdout.flush()
    rankings = pd.read_csv(rankings_path)
    print("Done")

    print("Fetching ranking... ", end=""); sys.stdout.flush()
    # Fetch all rankings by ID, then put 'em in a dict
    params = None
    ranking = {"predicted": {}, "benchmarked": {}}
    for i, row in rankings[rankings["ranking_id"] == ranking_id].iterrows():
        if params is None: params = tuple(row.iloc[2:8])
        elif tuple(row.iloc[2:8]) != params:
            print(f"ERROR: Parameters for row {i} are unexpected.", file=sys.stderr)
            return -1
        ranking["predicted"][row["ranking_position"]] = tuple(row.iloc[8:12])
        ranking["benchmarked"][row["ranking_position"]] = tuple(row.iloc[12:16])
    print("Done\n")

    print("Computing accuracy... ", end=""); sys.stdout.flush()
    # This is where the magic happens!
    # IDEA: Point-based system based on the distance each node has traveled. Not so much as an accuracy as a cost function.
    # Anything over 3 is not penalized anymore, leaving all nodes distance 3 at max penalty
    accuracy = 0
    total = 0
    for pi in range(len(ranking["predicted"])):
        predicted = ranking["predicted"][pi]
        
        # Find the matching one in the benchmarked list
        for bi in range(len(ranking["benchmarked"])):
            benchmarked = ranking["benchmarked"][bi]
            if predicted[:3] == benchmarked[:3]:
                # Add the difference to the list
                accuracy += abs(pi - bi)
                total += len(ranking["predicted"]) / 2
                break
    # Make it into an accuracy
    accuracy = (total - accuracy) / total * 100
    print("Done\n")

    line_size = 60
    print(f"Ranking {ranking_id}:")
    print("  Predicted:" + " " * (line_size - 6) + "Benchmarked:")
    for i in range(len(ranking["predicted"])):
        pm, pv, pt, pr = ranking["predicted"][i]
        bm, bv, bt, br = ranking["benchmarked"][i]

        text = f"   {i + 1}) {pm.upper()} ({pt} thread{'' if pt == 1 else 's'}) with {pv.upper()} ({pr:.4f}s)"
        text += " " * (line_size - len(text)) + " VS "
        text += f"   {i + 1}) {bm.upper()} ({bt} thread{'' if bt == 1 else 's'}) with {bv.upper()} ({br:.4f}s)"
        print(text)

    print(f"Accuracy: {accuracy:.2f}%")

    print("\nDone.\n")

    return 0


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--rankings", required=False, default=DEFAULT_RANKING_LOCATION, help="The location of the file containing the rankings.")
    parser.add_argument("-i", "--id", required=True, type=int, help="The ID of the ranking that we want to compare")

    args = parser.parse_args()

    # Check if rankings exist
    if not os.path.exists(args.rankings):
        print(f"ERROR: File '{args.rankings}' does not exist.", file=sys.stderr)
        exit(-1)
    if not os.path.isfile(args.rankings):
        print(f"ERROR: File '{args.rankings}' is not a file.", file=sys.stderr)
        exit(-1)

    exit(main(args.rankings, args.id))
