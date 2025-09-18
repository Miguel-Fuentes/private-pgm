import argparse
import time
import csv
import os

from mbi import Dataset
import ent_md
import max_ent
from scoring import mutual_information
import graph
import sampling
from itertools import combinations
import jax
import jax.numpy as jnp


def main(size_multiplier: float, max_iters: int, lambda_reg: float):
    # Load data and domain
    data = Dataset.load("../data/adult.csv", "../data/adult-domain.json")
    domain = data.domain
    total = data.df.shape[0]

    # Calculate mutual information scores for attribute pairs
    scores = {}
    margs = {}
    for comb in combinations(domain.attributes, 2):
        marg = data.project(comb)
        margs[comb] = marg
        scores[comb] = mutual_information(marg)

    # Number of samples for synthetic dataset
    n = int(total * size_multiplier)

    options = list(scores.keys())
    scores_array = jnp.array(list(scores.values()))

    # Start timing here - beginning of the actual synthesis process
    start_time = time.time()

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Get best permutation and MST edges
    perm = sampling.best_permutation(options, scores_array)
    mst_edges = graph.kruskal(perm, domain)
    order = graph.sampling_order(mst_edges)

    # Sample from the tree
    synth = sampling.sample_from_tree(margs, order, subkey, domain, n)

    # --- CORRECTED LOGIC ---
    # 1. Calculate MST error BEFORE removing duplicates
    n_margs = len(margs)
    scale_factor = 1 / size_multiplier
    avg_err_mst = 0
    for marg, true_value in margs.items():
        synth_marg_values = synth.project(marg).values
        scaled_synth_values = synth_marg_values * scale_factor
        mst_diff = scaled_synth_values - true_value.values
        avg_err_mst += jnp.linalg.norm(mst_diff, 1) / (n_margs * total)

    # 2. Now, remove duplicate records from synth before boosting and count them
    initial_synth_rows = synth.df.shape[0]
    synth.df.drop_duplicates(inplace=True)
    duplicates_removed = initial_synth_rows - synth.df.shape[0]

    # 3. Boosted entropies using the de-duplicated dataset
    cliques = list(combinations(domain.attributes, 2))
    ys = [data.project(clique).datavector() for clique in cliques]
    boosted = max_ent.public_support(synth, cliques, [1.0] * len(cliques), ys,
                                     max_iters=max_iters, lambda_reg=lambda_reg)

    # Systematic sampling and integer dataset creation
    integer_df = ent_md.systematic_sample(boosted.df, boosted.weights)
    boosted_integer = Dataset(df=integer_df, domain=domain)

    # End timing here - after all synthesis steps
    elapsed_time = time.time() - start_time

    # 4. Calculate remaining errors (Boosted and Integer)
    avg_err_boosted = 0
    avg_err_integer = 0

    for marg, true_value in margs.items():
        # Boosted error
        boosted_diff = boosted.project(marg).values - true_value.values
        avg_err_boosted += jnp.linalg.norm(boosted_diff, 1) / (n_margs * total)

        # Integer error
        integer_diff = boosted_integer.project(marg).values - true_value.values
        avg_err_integer += jnp.linalg.norm(integer_diff, 1) / (n_margs * total)

    # --- Modified CSV Writing Logic ---
    output_filename = "mst_boost.csv"
    
    # Check if the file exists before opening it
    file_exists = os.path.exists(output_filename)

    # Open the file in append mode. This will create it if it doesn't exist.
    with open(output_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # If the file did not exist, write the header row with the new column
        if not file_exists:
            writer.writerow(["size_multiplier", "max_iters", "reg", "mst_error", "boosted_error", "integer_error", "time_seconds", "duplicates_removed"])
        
        # Always write the data row for the current job, including the new value
        writer.writerow([size_multiplier, max_iters, lambda_reg, float(avg_err_mst), float(avg_err_boosted), float(avg_err_integer), elapsed_time, duplicates_removed])

    print(f"Results written to {output_filename}")
    print(f"Size multiplier: {size_multiplier}")
    print(f"Max iterations: {max_iters}")
    print(f"Lambda Reg: {lambda_reg}")
    print(f"Duplicates Removed: {duplicates_removed}")
    print(f"MST error: {avg_err_mst}")
    print(f"Boosted error: {avg_err_boosted}")
    print(f"Integer error: {avg_err_integer}")
    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and evaluate synthetic datasets using MST and boosting")
    parser.add_argument(
        "size_multiplier",
        type=float,
        help="Size of synthetic dataset as a multiple of the original dataset size (e.g., 0.25 for 25%)"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=5_000,
        help="Maximum iterations for the public support function (default: 5000)"
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=1e-6,
        help="Regularization parameter lambda for the public support function (default: 1e-6)"
    )
    args = parser.parse_args()
    main(args.size_multiplier, args.max_iters, args.lambda_reg)

