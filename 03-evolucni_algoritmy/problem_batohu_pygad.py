import numpy as np
import pygad
import matplotlib.pyplot as plt
import time

# Knapsack problem definition
weights = [10, 5, 8, 15, 4, 20, 7, 13, 9, 12]  # Weight of each item
values = [15, 8, 12, 25, 6, 35, 10, 20, 13, 18]  # Value of each item
max_weight = 60  # Maximum capacity of the knapsack

# Parameters
num_generations = 100
sol_per_pop = 50
num_parents_mating = 25
num_genes = len(weights)

# Fitness function
def fitness_func(ga_instance, solution, solution_idx):
    total_weight = np.sum(solution * weights)
    if total_weight > max_weight:
        return 0  # Penalize solutions that exceed max weight
    total_value = np.sum(solution * values)
    return total_value

# Run experiment with different operators and track progress
def run_experiment(selection_type, crossover_type, mutation_type, 
                   mutation_probability=0.1, title_suffix=""):
    # Create GA instance
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_type=int,
        gene_space=[0, 1],  # Binary representation (item is either included or not)
        parent_selection_type=selection_type,
        crossover_type=crossover_type,
        crossover_probability=0.8,
        mutation_type=mutation_type,
        mutation_probability=mutation_probability,
        keep_parents=-1,  # Keep all parents in the next population
    )
    
    # Start timer
    start_time = time.time()
    
    # Run the GA
    ga_instance.run()
    
    # End timer
    execution_time = time.time() - start_time
    
    # Get best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    # Calculate actual weight and value
    actual_weight = np.sum(solution * weights)
    actual_value = np.sum(solution * values)
    
    # Output information
    print(f"Experiment: {selection_type} + {crossover_type} + {mutation_type}")
    print(f"Best solution: {solution}")
    print(f"Fitness value: {solution_fitness}")
    print(f"Weight: {actual_weight}/{max_weight}")
    print(f"Value: {actual_value}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print("-" * 50)
    
    # Plot fitness progress
    plt.figure(figsize=(10, 6))
    plt.plot(ga_instance.best_solutions_fitness)
    plt.title(f"Fitness Progress\n{selection_type} + {crossover_type} + {mutation_type} {title_suffix}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    
    # Save the plot
    filename = f"fitness_{selection_type}_{crossover_type}_{mutation_type}.png".replace(" ", "_")
    plt.savefig(filename)
    plt.close()
    
    return ga_instance.best_solutions_fitness, solution_fitness, execution_time

# Compare different selection operators
def compare_selection_operators():
    crossover_type = "single_point"
    mutation_type = "random"
    
    # Run experiments with different selection types
    selection_types = ["sss", "rws", "tournament", "rank"]
    results = {}
    
    for selection in selection_types:
        fitness_history, best_fitness, exec_time = run_experiment(
            selection, crossover_type, mutation_type, 
            title_suffix="(Selection Comparison)"
        )
        results[selection] = (fitness_history, best_fitness, exec_time)
    
    # Plot comparison
    plt.figure(figsize=(12, 7))
    for selection, (history, _, _) in results.items():
        plt.plot(history, label=selection)
    
    plt.title("Selection Operators Comparison")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.savefig("selection_comparison.png")
    plt.close()
    
    # Print summary
    print("\nSelection Operators Summary:")
    print("-" * 30)
    for selection, (_, best_fitness, exec_time) in results.items():
        print(f"{selection}: Best Fitness = {best_fitness}, Time = {exec_time:.2f}s")

# Compare different crossover operators
def compare_crossover_operators():
    selection_type = "sss"  # Use steady-state selection
    mutation_type = "random"
    
    # Run experiments with different crossover types
    crossover_types = ["single_point", "two_points", "uniform", "scattered"]
    results = {}
    
    for crossover in crossover_types:
        fitness_history, best_fitness, exec_time = run_experiment(
            selection_type, crossover, mutation_type,
            title_suffix="(Crossover Comparison)"
        )
        results[crossover] = (fitness_history, best_fitness, exec_time)
    
    # Plot comparison
    plt.figure(figsize=(12, 7))
    for crossover, (history, _, _) in results.items():
        plt.plot(history, label=crossover)
    
    plt.title("Crossover Operators Comparison")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.savefig("crossover_comparison.png")
    plt.close()
    
    # Print summary
    print("\nCrossover Operators Summary:")
    print("-" * 30)
    for crossover, (_, best_fitness, exec_time) in results.items():
        print(f"{crossover}: Best Fitness = {best_fitness}, Time = {exec_time:.2f}s")

# Compare different mutation operators
def compare_mutation_operators():
    selection_type = "sss"
    crossover_type = "single_point"
    
    # Run experiments with different mutation types
    mutation_types = ["random", "swap", "inversion", "scramble"]
    results = {}
    
    for mutation in mutation_types:
        fitness_history, best_fitness, exec_time = run_experiment(
            selection_type, crossover_type, mutation,
            title_suffix="(Mutation Comparison)"
        )
        results[mutation] = (fitness_history, best_fitness, exec_time)
    
    # Plot comparison
    plt.figure(figsize=(12, 7))
    for mutation, (history, _, _) in results.items():
        plt.plot(history, label=mutation)
    
    plt.title("Mutation Operators Comparison")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.savefig("mutation_comparison.png")
    plt.close()
    
    # Print summary
    print("\nMutation Operators Summary:")
    print("-" * 30)
    for mutation, (_, best_fitness, exec_time) in results.items():
        print(f"{mutation}: Best Fitness = {best_fitness}, Time = {exec_time:.2f}s")

# Run the best combinations (based on previous experiments)
def run_best_combinations():
    # Define combinations to test (selection, crossover, mutation)
    combinations = [
        ("sss", "two_points", "random"),
        ("rank", "uniform", "swap"),
        ("tournament", "single_point", "inversion"),
        ("rws", "scattered", "scramble")
    ]
    
    results = {}
    
    for selection, crossover, mutation in combinations:
        combo_name = f"{selection}_{crossover}_{mutation}"
        fitness_history, best_fitness, exec_time = run_experiment(
            selection, crossover, mutation,
            title_suffix="(Best Combinations)"
        )
        results[combo_name] = (fitness_history, best_fitness, exec_time)
    
    # Plot comparison
    plt.figure(figsize=(12, 7))
    for combo, (history, _, _) in results.items():
        plt.plot(history, label=combo)
    
    plt.title("Best Operator Combinations Comparison")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.savefig("best_combinations.png")
    plt.close()
    
    # Print summary
    print("\nBest Combinations Summary:")
    print("-" * 30)
    for combo, (_, best_fitness, exec_time) in results.items():
        print(f"{combo}: Best Fitness = {best_fitness}, Time = {exec_time:.2f}s")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all comparisons
    print("Starting selection operators comparison...")
    compare_selection_operators()
    
    print("\nStarting crossover operators comparison...")
    compare_crossover_operators()
    
    print("\nStarting mutation operators comparison...")
    compare_mutation_operators()
    
    print("\nRunning best combinations...")
    run_best_combinations()
    
    print("\nAll experiments completed!")
