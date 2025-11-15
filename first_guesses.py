import numpy as np
from genetic_algorithm import create_random_allocation
import pandas as pd

GEN_DATA_PATH = "Data/GenerationData_B.csv"
LOAD_DATA_PATH = "Data/LoadData_B.csv"
NUM_PROFILES = 17

def create_smart_seed(ranked_pairs, ranked_buses):
    """
    Creates a "smart" seed allocation by assigning the best load-generation pairs
    to the weakest buses in the network.
    
    :param ranked_pairs: List of tuples (load_index, gen_index) ranked from best to worst
    :param ranked_buses: List of bus indices ranked from weakest to strongest
    :return: List mapping bus index to (load_index, gen_index)
    """
    print("Creating smart seed allocation...")

    load_allocation = [None] * len(ranked_buses)
    gen_allocation = [None] * len(ranked_buses)

    # Iterate NUM_PROFILES times, matchin best-to-weakest
    for i in range(len(ranked_pairs)):
        # Get the i-th best pair
        pair = ranked_pairs[i]

        # Get the i-th weakest bus
        location_index = ranked_buses[i]

        # Assign the load and generation indices to this bus
        load_allocation[location_index] = pair[0]
        gen_allocation[location_index] = pair[1]

    # Create final allocation tuple list
    allocation = (load_allocation, gen_allocation)
    print("Smart seed allocation created.")
    print("Load Allocation:", load_allocation)
    print("Generation Allocation:", gen_allocation)
    return allocation


def create_round_robin_seed(gen_data_path=GEN_DATA_PATH,
                            load_data_path=LOAD_DATA_PATH,
                            num_profiles=NUM_PROFILES
                            ) -> tuple[list[int], list[int]]: 
    """
    Construct a greedy round-robin seed allocation.
    Strategy:
    - Compute a simple exporter score per household (peak generation minus load).
    - Sort households by descending exporter score (big exporters first).
    - Spread them across bus positions using an even-odd interleaving (0,2,4...,1,3,5...) to avoid clustering.
    Returns a tuple (load_order, gen_order) where both lists assign the same household to the same bus.
    """

    if gen_data_path is None or load_data_path is None:
        # Fallback to random allocation if data not available
        return create_random_allocation()

    # Compute net export time series (generation - load) and take peak export per household
    net_export = (pd.read_csv(gen_data_path, index_col=0) - pd.read_csv(load_data_path, index_col=0))
    # peak export magnitude per household (MW)
    peak_export = net_export.max(axis=0)

    # Sort households by peak export descending; columns are assumed to be 0..NUM_PROFILES-1
    try:
        households = list(peak_export.sort_values(ascending=False).index.astype(int))
    except Exception:
        # Fallback: use column positions
        households = list(range(num_profiles))

    # Position order: interleave evens then odds to spread similar households
    pos_order = list(range(0, num_profiles, 2)) + list(range(1, num_profiles, 2))

    allocation = [None] * num_profiles
    for i, h in enumerate(households):
        pos = pos_order[i]
        allocation[pos] = int(h)# type: ignore

    # If any slots remain (shouldn't), fill them with remaining household indices
    remaining = [h for h in range(num_profiles) if h not in allocation]
    for i in range(num_profiles):
        if allocation[i] is None:
            allocation[i] = remaining.pop(0)# type: ignore

    # Use same assignment for loads and gens (household i assigned to bus allocation.index(i))
    return (allocation.copy(), allocation.copy())# type: ignore