
import random

items = [(2, 3), (3, 4), (4, 5), (5, 8), (9, 10)]  
max_weight = 10
pop_size = 10
mutation_rate = 0.1
num_generations = 50

def fitness(individual):
    total_weight = sum(ind * items[i][0] for i, ind in enumerate(individual))
    total_value = sum(ind * items[i][1] for i, ind in enumerate(individual))
    return total_value if total_weight <= max_weight else 0

def create_individual():
    return [random.randint(0, 1) for _ in range(len(items))]

def crossover(parent1, parent2):
    point = random.randint(1, len(items) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]

def select(population):
    return random.choices(population, weights=[fitness(ind) for ind in population], k=2)

def genetic_algorithm():
    population = [create_individual() for _ in range(pop_size)]
    for _ in range(num_generations):
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select(population)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = sorted(new_population, key=fitness, reverse=True)[:pop_size]
    best_solution = max(population, key=fitness)
    return best_solution, fitness(best_solution)

best, value = genetic_algorithm()
print(f'Nejlepší řešení: {best}, Hodnota: {value}')
