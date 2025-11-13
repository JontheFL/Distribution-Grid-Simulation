"""
Generic Algorithm for Distribution Grid Optimization

This script:
1. Defines a pandapower distribution grid (Grid B).
2. Implements a genetic algorithm to optimize the allocation of load and generation profiles
    to minimize the sum of maximum line loadings over a time series simulation.


Usage:
    python genetic_algorithm.py --num_pop 10 --num_gen 20
"""

import random
import numpy as np
import pandas as pd
import pandapower as pp
import sys, os
import uuid 
sys.path.insert(0, os.path.abspath('..'))
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

# --- GA Configuration ---
# "tune" these parameters to get better results
NUM_PROFILES = 17           # DO NOT CHANGE: number of load/generation nodes
POPULATION_SIZE = 12         # number of allocations to test in each generation
NUM_GENERATIONS = 5         # number of times to evolve the population
TOURNAMENT_SIZE = 3          # how many allocations "compete" to be a parent
MUTATION_RATE = 0.1          # 5% chance an allocation will be randomly mutated
CROSSOVER_RATE = 0.9         # 90% chance parents will "breed" (vs. just cloning)

# --- Global Data (loaded once) ---
# load the load and generation profiles
try: 
    GEN_DATA_ORIGINAL = pd.read_csv("Data/GenerationData_B.csv", index_col=0)
    LOAD_DATA_ORIGINAL = pd.read_csv("Data/LoadData_B.csv", index_col=0)
except FileNotFoundError:
    print("ERROR: Data files not found.")
    GEN_DATA_ORIGINAL = None
    LOAD_DATA_ORIGINAL = None

# --- 1. Pandapower Grid & Simulation ---
def create_grid_B():
    """
    Creates the pandapower network for Grid B.
    """
    # create empty net
    net = pp.create_empty_network()

    # create buses 0..17 (bus 0 is 20 kV, the rest are 0.4 kV)
    buses = [pp.create_bus(net, vn_kv=20.0 if i==0 else 0.4, name=f"Bus {i}") for i in range(18)]

    # create bus elements (use the first few buses as examples)
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.02, name="Grid Connection")
    for i in range(1, 18):
        pp.create_load(net, bus=buses[i], p_mw=0.1, q_mvar=0.005, name=f"Load {i}")
        pp.create_sgen(net, bus=buses[i], p_mw=0.1, q_mvar=0.005, name=f"Sgen {i}")


    # create branch elements
    tid = pp.create_transformer(
        net, hv_bus=buses[0], lv_bus=buses[1], std_type="0.4 MVA 20/0.4 kV", name="Trafo"
    )
    # create primary branch: buses 1 -> 2 -> ... -> 8
    for i in range(1, 8):
        pp.create_line(net, from_bus=buses[i], to_bus=buses[i+1], length_km=0.100, name=f"Line {i}-{i+1}", std_type="NAYY 4x50 SE")

    # create second branch: connect bus 1 to bus 9, then 9 -> 10 -> ... -> 17
    pp.create_line(net, from_bus=buses[1], to_bus=buses[9], length_km=0.100, name="Line 1-9", std_type="NAYY 4x50 SE")
    for i in range(9, 17):
        pp.create_line(net, from_bus=buses[i], to_bus=buses[i+1], length_km=0.100, name=f"Line {i}-{i+1}", std_type="NAYY 4x50 SE")
    
    return net

def get_base_indices(net) -> tuple[list[int], list[int]]: 
    """
    Get the order of node indices to be optimized.
    Assumes they are indexed 0 to 16.
    """
    index_order_load = list(net.load.index)
    index_order_gen = list(net.sgen.index)

    if len(index_order_load) != NUM_PROFILES or len(index_order_gen) != NUM_PROFILES:
        print(f"ERROR: Grid must have exactly {NUM_PROFILES} load/generation nodes.")
    
    return tuple([index_order_load, index_order_gen])

# --- 2. Fitness Function ---
def calculate_fitness(allocation: tuple[list[int], list[int]], net) -> tuple[float, float]:
    """
    This function takes an allocation and returns its
    fitness score (1 / sum_max_loading) and the sum_max_loading values.
    """
    try: 
        # 1. Call the time series simulation
        from utils import run_time_series

        # Create a folder for this simulation's results (for parallel runs later)
        res_suffix = f"ga_{uuid.uuid4()}"
        res_path = f"./results_{res_suffix}/"
       
        res_ext, res_lines = run_time_series(
            GEN_DATA_ORIGINAL,
            LOAD_DATA_ORIGINAL,
            net,
            index_order_load=allocation[0],
            index_order_gen=allocation[1],
            results_suffix=res_suffix,
        )

        # 2. Read the results
        res_file = os.path.join(res_path, "res_line", "loading_percent.csv")
        res_df = pd.read_csv(res_file, skiprows=1,sep=";")
        
        # 3. Calculate the sum of maximum line loadings for each timestep
        # max_loading = res_df.iloc[:, 1].max()
        sum_max_loading = res_lines.max(axis=1).sum() 

        # 4. Confirm that no line is overloaded (> 100%)
        if not all(res_lines.max(axis=1) <= 100):
            print(f"WARNING: Allocation {allocation} caused overload!")
            return 0.0, 99999.0 # Return very bad fitness and high loading
            
        # 5. Clean up the results folder
        try:
            import shutil
            shutil.rmtree(res_path)
        except Exception as e:
            print(f"ERROR: Could not delete results folder {res_path}: {e}")

        # 6. Calculate fitness (minimize loading -> maximize fitness)
        if sum_max_loading == 0:
            fitness = float('inf') # Avoid divistion by zero
        else:
            fitness = 1.0 / sum_max_loading

        return fitness, sum_max_loading
    
    except Exception as e:
        print(f"ERROR during fitness calculation: {e}")
        # If simulation fails, it's a "bad" allocation
        return 0.0, 99999.0 # Return very bad fitness and high loading

# --- 3. Genetic Algorithm Operators ---
def create_random_allocation() -> tuple[list[int], list[int]]:
    """
    Creates a random allocation of for load and gen.
    """
    index_order_load = random.sample(range(NUM_PROFILES), NUM_PROFILES)
    index_order_gen = random.sample(range(NUM_PROFILES), NUM_PROFILES)
    return tuple([index_order_load, index_order_gen])


def construct_round_robin_seed() -> tuple[list[int], list[int]]:
    """
    Construct a greedy round-robin seed allocation.
    Strategy:
    - Compute a simple exporter score per household (peak generation minus load).
    - Sort households by descending exporter score (big exporters first).
    - Spread them across bus positions using an even-odd interleaving (0,2,4...,1,3,5...) to avoid clustering.
    Returns a tuple (load_order, gen_order) where both lists assign the same household to the same bus.
    """
    if GEN_DATA_ORIGINAL is None or LOAD_DATA_ORIGINAL is None:
        # Fallback to random allocation if data not available
        return create_random_allocation()

    # Compute net export time series (generation - load) and take peak export per household
    net_export = (GEN_DATA_ORIGINAL - LOAD_DATA_ORIGINAL)
    # peak export magnitude per household (MW)
    peak_export = net_export.max(axis=0)

    # Sort households by peak export descending; columns are assumed to be 0..NUM_PROFILES-1
    try:
        households = list(peak_export.sort_values(ascending=False).index.astype(int))
    except Exception:
        # Fallback: use column positions
        households = list(range(NUM_PROFILES))

    # Position order: interleave evens then odds to spread similar households
    pos_order = list(range(0, NUM_PROFILES, 2)) + list(range(1, NUM_PROFILES, 2))

    allocation = [None] * NUM_PROFILES
    for i, h in enumerate(households):
        pos = pos_order[i]
        allocation[pos] = int(h)

    # If any slots remain (shouldn't), fill them with remaining household indices
    remaining = [h for h in range(NUM_PROFILES) if h not in allocation]
    for i in range(NUM_PROFILES):
        if allocation[i] is None:
            allocation[i] = remaining.pop(0)

    # Use same assignment for loads and gens (household i assigned to bus allocation.index(i))
    return (allocation.copy(), allocation.copy())

def create_initial_population(base_allocation: tuple[list[int], list[int]]) -> list[tuple[list[int], list[int]]]:
    """
    Creates the initial population of allocations.
    """
    # Start with the base allocation (keep original behavior without deepcopy)
    population: list[tuple[list[int], list[int]]] = [base_allocation]

    # Add a greedy round-robin seed to give the GA a smarter starting point
    try:
        greedy_seed = construct_round_robin_seed()
        population.append(greedy_seed)
    except Exception:
        # If anything goes wrong, fall back to a random allocation
        population.append(create_random_allocation())

    # Fill the remainder with random allocations
    for _ in range(POPULATION_SIZE - len(population)):
        population.append(create_random_allocation())
    return population

def tournament_selection(population: list[tuple[list[int], list[int]]], fitnesses: list[float]) -> tuple[list[int], list[int]]:
    """
    Selects a parent allocation using tournament selection.
    1. Pick TOURNAMENT_SIZE random allocations from the population.
    2. Find the one with the highest fitness.
    3. Return that allocation as the selected parent.
    """
    # Select random indices for the tournament (use actual population length)
    tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)

    # Get the allocations and their fitnesses
    tournament_allocations = [population[i] for i in tournament_indices]
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

    # Find the index of the winner (highest fitness)
    winner_index = np.argmax(tournament_fitnesses)
    return tournament_allocations[winner_index]

def ordered_crossover(parent1_tuple: tuple[list[int], list[int]], parent2_tuple: tuple[list[int], list[int]]) -> tuple[list[int], list[int]]:
    """
    "Breeds" two parents allocations to create a child, preserving the permutation.
    This is called "Ordered Crossover".

    1. Take a random "slice" from parent1.
    2. Fill in the child's other slots using genes from parent2, in order, skipping duplicates.
    """
    none_list = [None] * NUM_PROFILES
    child_tuple = (none_list.copy(), none_list.copy())

    # 1. Select two random crossover points
    start, end = sorted(random.sample(range(NUM_PROFILES), 2))

    # 2. Copy the "slice" from parent1 to the child
    child_tuple[0][start:end] = parent1_tuple[0][start:end]
    child_tuple[1][start:end] = parent1_tuple[1][start:end]

    # 3. Get the list of items not in the slice, in order from parent2
    parent2_items_load = [item for item in parent2_tuple[0] if item not in child_tuple[0]]
    parent2_items_gen = [item for item in parent2_tuple[1] if item not in child_tuple[1]]

    # 4. Fill the remaining 'None' slots for the child
    
    # for load allocation
    idx = 0
    for i in range(NUM_PROFILES):
        if child_tuple[0][i] is None:
            child_tuple[0][i] = parent2_items_load[idx]
            idx += 1
    
    idx = 0
    # for generation allocation
    for i in range(NUM_PROFILES):
        if child_tuple[1][i] is None:
            child_tuple[1][i] = parent2_items_gen[idx]
            idx += 1

    return child_tuple

def swap_mutation(allocation: tuple[list[int], list[int]]) -> tuple[list[int], list[int]]:
    """
    Introduces a random change in an allocation.
    1. Pick two random indices.
    2. Swap the values at those indices.
    """
    # Operate on copies to avoid mutating the parent allocation in-place
    idx1, idx2 = random.sample(range(NUM_PROFILES), 2)

    new_load = allocation[0].copy()
    new_gen = allocation[1].copy()

    # Swap the values at those indices on the copies
    new_load[idx1], new_load[idx2] = new_load[idx2], new_load[idx1]
    new_gen[idx1], new_gen[idx2] = new_gen[idx2], new_gen[idx1]

    return (new_load, new_gen)

def eval_allocation_wrapper(allocation: tuple[list[int], list[int]], calculate_fitness, net) -> tuple[float, float]:
    """
    This is a wrapper function. The 'multiprocessing.Pool' 
    can only pass one argument to its worker function.
    This wrapper takes that one argument and then calls out real 
    fitness function with all the other aruments it needs.
    """
    return calculate_fitness(allocation, net)

def main(): 
    global POPULATION_SIZE, NUM_GENERATIONS

    # --- Define Arguments Parser ---
    parser = argparse.ArgumentParser(description="A script to run genetic algorithm for distribution grid optimization.")

    # Add arguments for population size and number of generations
    parser.add_argument("--num_pop",
                        type=int,
                        default=POPULATION_SIZE,
                        help=f"Number of allocations in each generation. Default is {POPULATION_SIZE}."
                        )
    

    parser.add_argument("--num_gen",
                        type=int,
                        default=NUM_GENERATIONS,
                        help=f"Number of generations to run the algorithm. Default is {NUM_GENERATIONS}."
                        )

    args = parser.parse_args()

    # Update global GA parameters based on parsed arguments
    POPULATION_SIZE = args.num_pop
    NUM_GENERATIONS = args.num_gen

    # --- Test the fitness function ---
    print("--- Distribution Grid Genetic Algorithm Optimization Test ---")
    net = create_grid_B()
    print("Distribution Grid:\n", net)

    # Testing for base allocation and random allocation
    base_allocation = get_base_indices(net)
    print("\nBase Allocation Test:")
    print("Load Allocation:", base_allocation[0])
    print("Generation Allocation:", base_allocation[1])
    print(base_allocation)
    fitness, sum_max_loading = calculate_fitness(base_allocation, net)
    print(f"Fitness: {fitness}, Sum Max Loading: {sum_max_loading}")

    print("\nRandom Allocation Test:")
    rand_allocation = create_random_allocation()
    print("Load Allocation:", rand_allocation[0])
    print("Generation Allocation:", rand_allocation[1])
    fitness, sum_max_loading = calculate_fitness(rand_allocation, net)
    print(f"Fitness: {fitness}, Sum Max Loading: {sum_max_loading}")

    
    # Genetic Algorithm Execution
    print("\n========================================================")
    print("=== Distribution Grid Genetic Algorithm Optimization ===")
    print("========================================================")
    print(f"CPUs available: {cpu_count()}")
    print(f"Using {POPULATION_SIZE} allocations, evolving for {NUM_GENERATIONS} generations.")

    # --- 1. Setup ---
    # Track the solution found across ALL gnerations
    overall_best_allocation = None
    overall_best_loading = float('inf')

    # This 'partial' function "freezes" the extra arguments for our wrapper
    # This allows us to use 'Pool.map'
    eval_wrapper_with_args = partial(
        eval_allocation_wrapper, 
        calculate_fitness=calculate_fitness,
        net=net
    )

    # --- 2. Create Initial Population ---
    print("Creating initial population...")
    # Use the base allocation as one member of the initial population
    population = create_initial_population(base_allocation)

    # Diagnostic: show the first few members of the initial population (base, greedy seed, ...)
    try:
        print("Initial population preview (first 3):")
        for i, alloc in enumerate(population[:3]):
            print(f"  member {i}: Load order: {alloc[0]}\n            Gen order: {alloc[1]}")

        # If a greedy seed is present at index 1, evaluate and print its fitness
        if len(population) > 1:
            print("Evaluating greedy seed (initial population member 1):")
            g_fitness, g_sum = calculate_fitness(population[1], net)
            print(f"Greedy seed fitness: {g_fitness}, Sum Max Loading: {g_sum}")
    except Exception as e:
        print(f"Warning: could not display/evaluate initial population preview: {e}")

    # --- 3. Run the Evolution Loop ---
    for gen in range(NUM_GENERATIONS):
        print(f"\n--- Generation {gen + 1} / {NUM_GENERATIONS} ---")

        # --- 3a. Evaluate Fitness ---
        print(f"Evaluationg {len(population)} allocations...")

        # This 'with' block creates a pool of worker processes
        # It automaticalls maps the 'eval_wrapper_with_args' function to each allocation
        # in the 'population' list across multiple CPU cores
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(eval_wrapper_with_args, population)

        # Later : parallelize this step
        # results = [calculate_fitness(allocation, net) for allocation in population]

        fitnesses = [res[0] for res in results]
        loadings = [res[1] for res in results]

        # --- 3b. Find and report the best of this generation ---  
        best_gen_fitness = max(fitnesses)
        best_gen_index = fitnesses.index(best_gen_fitness)
        best_gen_allocation = population[best_gen_index]
        best_gen_loading = loadings[best_gen_index]

        print(f"Best Allocation: {best_gen_allocation}, Fitness: {best_gen_fitness}, Max Loading: {best_gen_loading}")

        # -- 3c. Update overall best solution ---
        if best_gen_loading < overall_best_loading:
            overall_best_loading = best_gen_loading
            overall_best_allocation = best_gen_allocation
            print(f"***New overall best allocation found! Max Loading: {overall_best_loading} ***")

        # --- 3d. Create the next generation ---
        new_population = []

        # Elitism: carry over the best allocation from this generation
        new_population.append(best_gen_allocation)

        while len(new_population) < POPULATION_SIZE:
            # 1. Selection - select two parents
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # 2. Crossover - breed the parents to create a child
            if random.random() < CROSSOVER_RATE:
                child = ordered_crossover(parent1, parent2)
            else:
                # 3. No crossover, just clone parent1
                child = parent1

            # 4. Mutation - randomly mutate the child
            if random.random() < MUTATION_RATE:
                child = swap_mutation(child)

            # 5. Add the child to the new population
            new_population.append(child)
        
        # The new population becomes the current population
        population = new_population

    # --- 4. Report the overall best solution found ---
    print("\n===============================================")
    print("=== Genetic Algorithm Optimization Complete ===")
    print("===============================================")
    print(f"Final sum of max loadings: {overall_best_loading:.2f}%")
    print(f"Load Allocation: {overall_best_allocation[0]}")
    print(f"Generation Allocation: {overall_best_allocation[1]}")
    

if __name__ == "__main__":
    main()