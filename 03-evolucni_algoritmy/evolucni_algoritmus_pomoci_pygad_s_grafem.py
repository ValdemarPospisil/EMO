import pygad
import numpy as np
import matplotlib.pyplot as plt

# Nastavení genetického algoritmu
num_generations = 50
num_parents_mating = 4

# Definice fitness funkce
def fitness_function(ga_instance ,solution, solution_idx):
    weights = [2, 3, 4, 5, 9]  # Váhy předmětů
    values = [3, 4, 5, 8, 10]  # Hodnoty předmětů
    max_weight = 10

    # Celková váha a hodnota řešení
    total_weight = np.sum(np.array(weights) * solution)
    total_value = np.sum(np.array(values) * solution)

    # Penalizace pokud váha překročí limit
    if total_weight > max_weight:
        return 0  # Diskvalifikace řešení
    return total_value

# Inicializace Pygad
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=10,
    num_genes=5,
    gene_space=[0, 1],  # Jen binární hodnoty 0 nebo 1
    init_range_low=0,
    init_range_high=2,
    mutation_percent_genes=10,
)

ga_instance.run()

# Získání historie fitness hodnot
fitness_history = ga_instance.best_solutions_fitness

plt.plot(fitness_history, marker='o', linestyle='-', color='b', label="Nejlepší fitness")
plt.xlabel("Generace")
plt.ylabel("Fitness hodnota")
plt.title("Vývoj nejlepší fitness hodnoty během generací")
plt.legend()
plt.grid()

plt.savefig("fitness_plot.png", dpi=300)  # Uložení do souboru s vysokým rozlišením
plt.show()
