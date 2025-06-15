import random

def blind_search(problem, max_iterations: int) -> tuple[list[int], int, list[int]]:
    best_solution = None
    best_fitness = -1
    fitness_history = []

    for _ in range(max_iterations):
        candidate_solution = problem.get_random_solution()
        candidate_fitness = problem.fitness(candidate_solution)

        if candidate_fitness > best_fitness:
            best_fitness = candidate_fitness
            best_solution = candidate_solution
        
        fitness_history.append(best_fitness)

    return best_solution, best_fitness, fitness_history


def get_neighbors(solution: list[int]) -> list[list[int]]:
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution[:]
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors

def hill_climbing(problem, max_iterations: int) -> tuple[list[int], int, list[int]]:

    current_solution = problem.get_random_solution()
    current_fitness = problem.fitness(current_solution)
    fitness_history = [current_fitness]

    for i in range(max_iterations):
        neighbors = get_neighbors(current_solution)
        best_neighbor_solution = None
        best_neighbor_fitness = -1

        for neighbor in neighbors:
            neighbor_fitness = problem.fitness(neighbor)
            if neighbor_fitness > best_neighbor_fitness:
                best_neighbor_fitness = neighbor_fitness
                best_neighbor_solution = neighbor

        if best_neighbor_fitness > current_fitness:
            current_solution = best_neighbor_solution
            current_fitness = best_neighbor_fitness
        else:
            fitness_history.extend([current_fitness] * (max_iterations - i - 1))
            break
        
        fitness_history.append(current_fitness)

    return current_solution, current_fitness, fitness_history
