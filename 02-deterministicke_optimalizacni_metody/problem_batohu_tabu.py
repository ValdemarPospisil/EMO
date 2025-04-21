import random
import matplotlib.pyplot as plt

# Parametry problému (stejné jako v předchozím úkolu)
values = [60, 100, 120, 80, 30, 50, 70, 20]
weights = [10, 20, 30, 15, 5, 8, 12, 3]
max_weight = 50
num_items = len(values)

# Výpočet fitness
def fitness(solution):
    total_weight = sum(w for w, s in zip(weights, solution) if s)
    total_value = sum(v for v, s in zip(values, solution) if s)
    return total_value if total_weight <= max_weight else 0

# Generuj náhodné binární řešení
def random_solution():
    return [random.randint(0, 1) for _ in range(num_items)]

# Generuj všechny sousedy (flip každého bitu)
def get_neighbors(solution):
    neighbors = []
    for i in range(num_items):
        neighbor = solution.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors

# Tabu Search
def tabu_search(max_iterations=100, tabu_size=10):
    current = random_solution()
    best = current
    tabu_list = []

    fitness_progress = [fitness(current)]

    for _ in range(max_iterations):
        neighbors = get_neighbors(current)
        valid_neighbors = [n for n in neighbors if n not in tabu_list]

        if not valid_neighbors:
            break

        next_candidate = max(valid_neighbors, key=fitness)

        if fitness(next_candidate) > fitness(best):
            best = next_candidate

        tabu_list.append(current)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        current = next_candidate
        fitness_progress.append(fitness(current))

    return best, fitness(best), fitness_progress

# --- Spuštění algoritmu ---
best_solution, best_fitness, fitness_progress = tabu_search()

print("Tabu Search:")
print(f"  Nejlepší řešení: {best_solution}")
print(f"  Fitness: {best_fitness}")

# --- Graf ---
plt.plot(fitness_progress)
plt.title("Tabu Search - Vývoj fitness")
plt.xlabel("Iterace")
plt.ylabel("Fitness")
plt.grid(True)
plt.savefig("fitness_progress_tabu.png")
plt.close()
