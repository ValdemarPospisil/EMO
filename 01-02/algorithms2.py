import random
import numpy as np
from collections import deque

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


def hill_climbing_with_learning(problem, max_iterations: int, pop_size: int, b_best: int, learning_rate: float) -> tuple[list[int], int, list[int]]:
    num_dims = problem.num_items
    
    w = np.full(num_dims, 0.5)

    overall_best_solution = None
    overall_best_fitness = -1
    fitness_history = []

    for _ in range(max_iterations):
        population = []
        for _ in range(pop_size):
            solution = (np.random.rand(num_dims) < w).astype(int).tolist()
            fitness = problem.fitness(solution)
            population.append({'solution': solution, 'fitness': fitness})
        
        population.sort(key=lambda x: x['fitness'], reverse=True)
        best_in_population = population[0]

        if best_in_population['fitness'] > overall_best_fitness:
            overall_best_fitness = best_in_population['fitness']
            overall_best_solution = best_in_population['solution']
        
        fitness_history.append(overall_best_fitness)

        b_best_solutions = [p['solution'] for p in population[:b_best]]
        
        if b_best > 0:
            avg_best_vector = np.mean(np.array(b_best_solutions), axis=0)
            w = w + learning_rate * (avg_best_vector - w)
            w = np.clip(w, 0.001, 0.999)

    return overall_best_solution, overall_best_fitness, fitness_history


def tabu_search(problem, max_iterations: int, tabu_tenure: int) -> tuple[list[int], int, list[int]]:
    current_solution = problem.get_random_solution()
    best_solution_so_far = current_solution
    best_fitness_so_far = problem.fitness(best_solution_so_far)
    fitness_history = [best_fitness_so_far]
    
    tabu_list = deque(maxlen=tabu_tenure)

    for _ in range(max_iterations):
        best_neighbor = None
        best_neighbor_fitness = -1
        best_move = -1

        for i in range(problem.num_items):
            neighbor = current_solution[:]
            neighbor[i] = 1 - neighbor[i]
            neighbor_fitness = problem.fitness(neighbor)

            is_tabu = i in tabu_list
            aspiration_criterion_met = neighbor_fitness > best_fitness_so_far
            
            if (not is_tabu or aspiration_criterion_met) and neighbor_fitness > best_neighbor_fitness:
                best_neighbor = neighbor
                best_neighbor_fitness = neighbor_fitness
                best_move = i
        
        if best_neighbor is not None:
            current_solution = best_neighbor
            
            tabu_list.append(best_move)

            if best_neighbor_fitness > best_fitness_so_far:
                best_solution_so_far = best_neighbor
                best_fitness_so_far = best_neighbor_fitness
        
        fitness_history.append(best_fitness_so_far)

    return best_solution_so_far, best_fitness_so_far, fitness_history
