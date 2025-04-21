import pygad
import numpy as np


items = [(2, 3), (3, 4), (4, 5), (5, 8), (9, 10), (2, 6), (5, 4), (4, 7), (3, 8)]
max_weight = 10
sol_per_pop = 10
num_generations = 50
mutation_rate = 0.1

def fitness_function(ga_instance, solution, solution_idx):
    total_weight = sum(solution[i] * items[i][0] for i in range(len(items)))
    total_value = sum(solution[i] * items[i][1] for i in range(len(items)))
    return total_value if total_weight <= max_weight else 0


ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=5,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=len(items),
    gene_type=int,
    init_range_low=0,
    init_range_high=2,
    parent_selection_type="tournament",
    keep_parents=1,
    crossover_type = "single_point",
    mutation_type = "random",
    mutation_percent_genes = int(mutation_rate * 100),
    mutation_num_genes=1
    )


ga_instance.run()


solution, solution_fitness, solution_idx = ga_instance.best_solution()

print(f'Nejlepší řešení: {solution}, Hodnota: {solution_fitness}')



