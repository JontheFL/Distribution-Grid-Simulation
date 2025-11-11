from random import random
import numpy as np
import pandas as pd
import pandapower as pp
import sys, os
import uuid 
sys.path.insert(0, os.path.abspath('..'))

# --- GA Configuration ---
# "tune" these parameters to get better results
NUM_PROFILES = 2            # DO NOT CHANGE: number of load/generation nodes
POPULATION_SIZE = 50         # number of allocations to test in each generation
NUM_GENERATIONS = 30         # number of times to evolve the population
TOURNAMENT_SIZE = 3          # how many allocations "compete" to be a parent
MUTATION_RATE = 0.1          # 10% chance an allocation will be randomly mutated
CROSSOVER_RATE = 0.9         # 90% chance parents will "breed" (vs. just cloning)

# --- Global Data (loaded once) ---
# load the load and generation profiles
try: 
    GEN_DATA_ORIGINAL = pd.read_csv("Data/GenerationData_A.csv", index_col=0)
    LOAD_DATA_ORIGINAL = pd.read_csv("Data/LoadData_A.csv", index_col=0)
except FileNotFoundError:
    print("ERROR: Data files not found.")
    GEN_DATA_ORIGINAL = None
    LOAD_DATA_ORIGINAL = None

# --- 1. Pandapower Grid & Simulation ---
def create_grid_B():
    """
    Creates the pandapower network for Grid B.
    """

    # WARNING: This is system A
    # create empty net
    net = pp.create_empty_network()

    # create buses
    b1 = pp.create_bus(net, vn_kv=20.0, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")

    # create bus elements
    pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b2, p_mw=0.1, q_mvar=0.005, name="Load")
    pp.create_load(net, bus=b3, p_mw=0.1, q_mvar=0.005, name="Load")
    pp.create_sgen(net, bus=b2, p_mw=0.1, q_mvar=0.005, name="Load")
    pp.create_sgen(net, bus=b3, p_mw=0.1, q_mvar=0.005, name="Load")

    # create branch elements
    tid = pp.create_transformer(
        net, hv_bus=b1, lv_bus=b2, std_type="0.4 MVA 20/0.4 kV", name="Trafo"
    )
    pp.create_line(
        net, from_bus=b2, to_bus=b3, length_km=0.100, name="Line", std_type="NAYY 4x50 SE"
    )
    
    return net

def get_base_indices(net):
    """
    Get the list of node indices to be optimized.
    Assumes they are indexed 0 to 16.
    """
    base_indices = list(net.load.index) # maybe change this line

    if len(base_indices) != NUM_PROFILES:
        print(f"ERROR: Grid must have exactly {NUM_PROFILES} load/generation nodes.")
    
    return base_indices

# --- 2. Fitness Function ---
def calculate_fitness(allocation: list[int], net) -> tuple[float, float]:
    """
    This function takes an allocation and returns its
    fitness score (1 / max_loading) and the max_loading values.
    """
    try: 
        # 1. Call the time series simulation
        from utils import run_time_series

        # Create a folder for this simulation's results (for parallel runs later)
        res_suffix = f"ga_{uuid.uuid4()}"
        res_path = f"./results_{res_suffix}/"
       
        run_time_series(
            GEN_DATA_ORIGINAL,
            LOAD_DATA_ORIGINAL,
            net,
            index_order_load=allocation,
            index_order_gen=allocation,
            results_suffix=res_suffix,
        )

        # 2. Read the results
        res_file = os.path.join(res_path, "res_line", "loading_percent.csv")
        res_df = pd.read_csv(res_file, skiprows=1,sep=";")
        
        # 3. Find the maximum line loading
        max_loading = res_df.iloc[:, 1].max()
    
        # 4. Clean up the results folder (MISSING)
        try:
            import shutil
            shutil.rmtree(res_path)
        except Exception as e:
            print(f"ERROR: Could not delete results folder {res_path}: {e}")

        # 5. Calculate fitness (minimize loading -> maximize fitness)
        if max_loading == 0:
            fitness = float('inf') # Avoid divistion by zero
        else:
            fitness = 1.0 / max_loading

        return fitness, max_loading
    
    except Exception as e:
        print(f"ERROR during fitness calculation: {e}")
        # If simulation fails, it's a "bad" allocation
        return 0.0, 99999.0 # Return very bad fitness and high loading

# --- 3. Genetic Algorithm Operators ---
def create_random_allocation() -> list[int]:
    """
    Creates a random allocation of profiles.
    """
    allocation = random.sample(range(NUM_PROFILES), NUM_PROFILES)
    return allocation

def create_initial_population() -> list[list[int]]:
    """
    Creates the initial population of allocations.
    """
    population = []
    for _ in range(POPULATION_SIZE):
        population.append(create_random_allocation())
    return population

def tournament_selectino(population: list[list[int]], fitnesses: list[float]) -> list[int]:
    """
    Selects a parent allocation using tournament selection.
    1. Pick TOURNAMENT_SIZE random allocations from the population.
    2. Find the one with the highest fitness.
    3. Return that allocation as the selected parent.
    """
    # Select random indices for the tournament
    tournament_indices = random.sample(range(POPULATION_SIZE), TOURNAMENT_SIZE)

    # Get the allocations and their fitnesses
    tournament_allocations = [population[i] for i in tournament_indices]
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

    # Find the index of the winner (highest fitness)
    winner_index = np.argmax(tournament_fitnesses)
    return tournament_allocations[winner_index]

def ordered_crossover(parent1: list[int], parent2: list[int]) -> list[int]:
    """
    "Breeds" two parents allocations to create a child, preserving the permutation.
    This is called "Ordered Crossover".

    1. Take a random "slice" from parent1.
    2. Fill in the child's other slots using genes from parent2, in order, skipping duplicates.
    """
    child = [None] * NUM_PROFILES

    # 1. Select two random crossover points
    start, end = sorted(random.sample(range(NUM_PROFILES), 2))

    # 2. Copy the "slice" from parent1 to the child
    child[start:end] = parent1[start:end]

    # 3. Get the list of items not in the slice, in order from parent2
    parent2_items = [item for item in parent2 if item not in child]

    # 4. Fill the remaining 'None' slots for the child
    idx = 0
    for i in range(NUM_PROFILES):
        if child[i] is None:
            child[i] = parent2_items[idx]
            idx += 1

    return child

def swap_mutation(allocation: list[int]) -> list[int]:
    """
    Introduces a random change in an allocation.
    1. Pick two random indices.
    2. Swap the values at those indices.
    """
    # Get two random indices to swap
    idx1, idx2 = random.sample(range(NUM_PROFILES), 2)

    # Swap the values at those indices
    allocation[idx1], allocation[idx2] = allocation[idx2], allocation[idx1]
    return allocation

def main(): 
    print("--- Distribution Grid Genetic Algorithm Optimization ---")
    allocation = [1, 0]
    net = create_grid_B()
    print(net)
    print("Base Indices:", get_base_indices(net))
    fitness, max_loading = calculate_fitness(allocation, net)
    print(f"Test Allocation: {allocation}, Fitness: {fitness}, Max Loading: {max_loading}")


if __name__ == "__main__":
    main()