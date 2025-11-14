import numpy as np

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

if __name__ == "__main__":
    # Example usage
    ranked_pairs = [(8, 14), (12, 15), (11, 10), (15, 8), (1, 3), (5, 6), (9, 5), (6, 2), (0, 11), (16, 7), (14, 9), (3, 1), (2, 13), (4, 4), (7, 0), (10, 16), (13, 12)]
    ranked_buses = [16, 15, 14, 7, 13, 6, 12, 5, 4, 11, 3, 10, 9, 2, 8, 1, 0]
    
    allocation = create_smart_seed(ranked_pairs, ranked_buses)