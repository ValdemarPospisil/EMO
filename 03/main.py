import pygad
import numpy as np
import matplotlib.pyplot as plt

ITEMS = [
    (10, 60), (12, 50), (20, 120), (5, 30), (15, 80), (8, 45), (4, 25),
    (18, 100), (2, 15), (7, 40), (11, 55), (22, 130), (6, 35), (9, 65),
    (13, 70), (1, 10), (16, 90), (19, 110), (3, 20), (14, 75)
]
item_weights = np.array([item[0] for item in ITEMS])
item_values = np.array([item[1] for item in ITEMS])
CAPACITY = 100
NUM_ITEMS = len(ITEMS)

def fitness_func(ga_instance, solution, solution_idx):
    total_weight = np.sum(solution * item_weights)
    total_value = np.sum(solution * item_values)

    if total_weight > CAPACITY:
        return 0
    else:
        return total_value

common_ga_params = {
    "num_generations": 100,
    "num_parents_mating": 10,
    "fitness_func": fitness_func,
    "sol_per_pop": 50,
    "num_genes": NUM_ITEMS,
    "gene_space": [0, 1], 
    "keep_elitism": 2, 
    "suppress_warnings": True
}

configurations = {
    "Konfigurace A (Klasická)": {
        "parent_selection_type": "sss",
        "crossover_type": "single_point",
        "mutation_type": "random",
        "mutation_percent_genes": 10,
        "color": "blue"
    },
    "Konfigurace B (Turnajová)": {
        "parent_selection_type": "tournament",
        "K_tournament": 3,
        "crossover_type": "two_points",
        "mutation_type": "random",
        "mutation_percent_genes": 10,
        "color": "green"
    },
    "Konfigurace C (Agresivní)": {
        "parent_selection_type": "rws",
        "crossover_type": "uniform",
        "mutation_type": "adaptive",
        "mutation_probability": [0.3, 0.1],
        "color": "red"
    }
}

def plot_comparison_graph(histories: dict, filename="ga_comparison.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    
    for name, data in histories.items():
        plt.plot(data['history'], label=name, color=data['color'])
    
    plt.title("Porovnání konfigurací Genetického Algoritmu", fontsize=16)
    plt.xlabel("Generace", fontsize=12)
    plt.ylabel("Nejlepší fitness v generaci", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"\nGraf byl uložen do souboru '{filename}'")
    plt.show()

def main():
    print("--- EMO: Úkol 3 - Genetické algoritmy (PyGAD) ---")
    
    all_histories = {}

    for name, config in configurations.items():
        print(f"\nSpouštím: {name}")
        
        # Spojíme společné a specifické parametry
        current_params = {**common_ga_params, **{k: v for k, v in config.items() if k != 'color'}}
        
        # Vytvoření a spuštění instance GA
        ga_instance = pygad.GA(**current_params)
        ga_instance.run()

        # Získání výsledků a historie
        solution, solution_fitness, _ = ga_instance.best_solution()
        print(f"   -> Nejlepší nalezené fitness: {solution_fitness:.2f}")

        # Uložení historie pro graf
        all_histories[name] = {
            'history': ga_instance.best_solutions_fitness,
            'color': config['color']
        }
        
    # Vykreslení srovnávacího grafu
    plot_comparison_graph(all_histories)

if __name__ == "__main__":
    main()
