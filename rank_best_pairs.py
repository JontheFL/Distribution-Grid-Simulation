import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys, os
from pathlib import Path

def rank_best_pairs(gen_data_path, load_data_path, num_profiles) -> list[tuple[int, int]]:
    """
    Finds and ranks the best load-generation pairs based on minimizing peak net load.
    """
    try: 
        # --- 0. Load Data ---
        gen_data = pd.read_csv(Path(gen_data_path), index_col=0)
        load_data = pd.read_csv(Path(load_data_path), index_col=0)
        # --- 1. Create the NxN cost matrix ---
        # Cost of pairing load i and gen j
        # Peak Net Load as cost metric
        # A lower cost is better
        cost_matrix = np.zeros((num_profiles, num_profiles))

        # Create DataFrames for easier coumn access
        load_df = load_data[[str(i) for i in range(num_profiles)]]
        gen_df = gen_data[[str(j) for j in range(num_profiles)]]

        for i in range(num_profiles):
            for j in range(num_profiles):
                # Calculate net load for specific load-gen pair
                net_load = load_df[str(i)] - gen_df[str(j)]

                # The cost is the single highest peak net load across all timesteps
                cost_matrix[i, j] = net_load.max()

        # --- 2. Solve the assignment problem ---
        # Find the best pairing
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # --- 3. Create the "Best Pairs" allocation ---
        best_pairs = list(zip(row_ind.tolist(), col_ind.tolist()))

        # --- 4. Rank thes pairs by their costs ---
        pair_costs = cost_matrix[row_ind, col_ind]

        # Get the indices that would sort the pairs from best to worst
        pair_ranking_indices = np.argsort(pair_costs)

        ranked_pairs_list = []
        for k in pair_ranking_indices:
            pair = best_pairs[k]
            cost = pair_costs[k]
            ranked_pairs_list.append((pair, cost))
            #print(f"{len(ranked_pairs_list)-1}: (L{pair[0]}, G{pair[1]}) (Cost: {cost:.6f})"

        # --- 6. Create list of tuples of best pairs ---
        pairing_list = []
        for k in pair_ranking_indices:
            pairing_list.append(best_pairs[k])

        return pairing_list
    except Exception as e:
        print("An error occurred:", e)

