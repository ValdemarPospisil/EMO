import random
import matplotlib.pyplot as plt

# Parametry problému batohu 
ITEMS = [
    {"value": 10, "weight": 5},
    {"value": 20, "weight": 7},
    {"value": 15, "weight": 6},
    {"value": 40, "weight": 15},
    {"value": 30, "weight": 10},
    {"value": 50, "weight": 20},
    {"value": 35, "weight": 18},
    {"value": 25, "weight": 12}
]
MAX_WEIGHT = 40
DIMENSION = len(ITEMS)

# Fitness funkce
def fitness(individual):
    total_value = 0
    total_weight = 0
    for bit, item in zip(individual, ITEMS):
        if bit:
            total_value += item["value"]
            total_weight += item["weight"]
    if total_weight > MAX_WEIGHT:
        return 0
    return total_value

# Generátor náhodného kandidáta
def random_candidate():
    return [random.randint(0, 1) for _ in range(DIMENSION)]

# Generování sousedů
def get_neighbors(individual):
    neighbors = []
    for i in range(DIMENSION):
        neighbor = individual.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors

# Slepý algoritmus
def blind_search(iterations):
    best = random_candidate()
    best_fitness = fitness(best)
    fitness_progress = [best_fitness]

    for _ in range(iterations):
        candidate = random_candidate()
        f = fitness(candidate)
        if f > best_fitness:
            best = candidate
            best_fitness = f
        fitness_progress.append(best_fitness)

    return best, best_fitness, fitness_progress

# Horolezecký algoritmus
def hill_climbing(iterations):
    current = random_candidate()
    current_fitness = fitness(current)
    fitness_progress = [current_fitness]

    for _ in range(iterations):
        neighbors = get_neighbors(current)
        neighbor_fitnesses = [(n, fitness(n)) for n in neighbors]
        best_neighbor, best_neighbor_fitness = max(neighbor_fitnesses, key=lambda x: x[1])

        if best_neighbor_fitness > current_fitness:
            current = best_neighbor
            current_fitness = best_neighbor_fitness
        fitness_progress.append(current_fitness)

    return current, current_fitness, fitness_progress

# Vizualizace
def plot_fitness(progress, title, filename):
    plt.figure()
    plt.plot(progress)
    plt.title(title)
    plt.xlabel("Iterace")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    iterations = 100

    # Slepý algoritmus
    blind_solution, blind_fitness, blind_progress = blind_search(iterations)
    print("Slepý algoritmus:")
    print("  Nejlepší řešení:", blind_solution)
    print("  Fitness:", blind_fitness)
    plot_fitness(blind_progress, "Slepý algoritmus - Vývoj fitness", "fitness_blind.png")

    # Horolezecký algoritmus
    hill_solution, hill_fitness, hill_progress = hill_climbing(iterations)
    print("Horolezecký algoritmus:")
    print("  Nejlepší řešení:", hill_solution)
    print("  Fitness:", hill_fitness)
    plot_fitness(hill_progress, "Horolezecký algoritmus - Vývoj fitness", "fitness_hillclimb.png")
