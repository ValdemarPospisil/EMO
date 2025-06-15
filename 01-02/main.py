import matplotlib.pyplot as plt
from knapsack import KnapsackProblem
from algorithms2 import blind_search, hill_climbing, hill_climbing_with_learning, tabu_search

ITEMS = [
    (10, 60), (12, 50), (20, 120), (5, 30), (15, 80), (8, 45), (4, 25),
    (18, 100), (2, 15), (7, 40), (11, 55), (22, 130), (6, 35), (9, 65),
    (13, 70), (1, 10), (16, 90), (19, 110), (3, 20), (14, 75)
]
CAPACITY = 100
MAX_ITERATIONS = 500

POP_SIZE = 50       # Kolik řešení generovat v každém kroku
B_BEST = 5          # Z kolika nejlepších se učit
LEARNING_RATE = 0.1 # Jak rychle se má pravděpodobnostní vektor měnit

TABU_TENURE = 7     # Jak dlouho si pamatovat zakázaný krok (typicky malé prvočíslo)


def plot_results(histories: dict, filename="fitness_comparison.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    
    for name, data in histories.items():
        plt.plot(data['history'], label=name, color=data['color'], linestyle=data.get('style', '-'), alpha=0.9)
    
    plt.title("Porovnání optimalizačních algoritmů pro problém batohu", fontsize=16)
    plt.xlabel("Iterace", fontsize=12)
    plt.ylabel("Nejlepší nalezená fitness (celková cena)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"\nGraf byl uložen do souboru '{filename}'")
    plt.show()

def main():
    print("--- EMO: Úkol 2 - Deterministické prohledávání ---")
    
    knapsack_problem = KnapsackProblem(items=ITEMS, capacity=CAPACITY)
    print(f"Řešíme problém: {knapsack_problem}")
    print(f"Počet iterací pro každý algoritmus: {MAX_ITERATIONS}\n")

    histories = {}

    # 1. Slepé prohledávání (baseline)
    print("1. Spouštím Slepé prohledávání...")
    _, bs_fit, bs_hist = blind_search(knapsack_problem, MAX_ITERATIONS)
    print(f"   -> Nejlepší fitness: {bs_fit}")
    histories['Slepé prohledávání'] = {'history': bs_hist, 'color': 'grey', 'style': ':'}

    # 2. Horolezec (Strmé stoupání)
    print("2. Spouštím Horolezce (Strmé stoupání)...")
    _, hc_fit, hc_hist = hill_climbing(knapsack_problem, MAX_ITERATIONS)
    print(f"   -> Fitness na vrcholu: {hc_fit}")
    histories['Horolezec (základní)'] = {'history': hc_hist, 'color': 'orange'}
    
    # 3. Horolezec s učením (HCwL)
    print("\n3. Spouštím Horolezce s učením (HCwL)...")
    _, hcwl_fit, hcwl_hist = hill_climbing_with_learning(
        knapsack_problem, MAX_ITERATIONS, POP_SIZE, B_BEST, LEARNING_RATE
    )
    print(f"   -> Nejlepší fitness: {hcwl_fit}")
    histories['Horolezec s učením'] = {'history': hcwl_hist, 'color': 'green'}

    # 4. Tabu Search
    print("4. Spouštím Tabu Search...")
    _, ts_fit, ts_hist = tabu_search(knapsack_problem, MAX_ITERATIONS, TABU_TENURE)
    print(f"   -> Nejlepší fitness: {ts_fit}")
    histories['Tabu Search'] = {'history': ts_hist, 'color': 'blue', 'style': '-'}

    plot_results(histories)

if __name__ == "__main__":
    main()
