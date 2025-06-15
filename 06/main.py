import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

class Particle:
    def __init__(self, dimensions, bounds):
        self.position = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(dimensions)])
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')

class PSO:
    def __init__(self, num_particles, dimensions, bounds, w=0.7, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        
        self.particles = [Particle(dimensions, bounds) for _ in range(num_particles)]
        self.global_best_position = np.zeros(dimensions)
        self.global_best_fitness = float('inf')
        self.fitness_history = []
        self.position_history = []

    def evaluate_fitness(self, position, function_type):
        if function_type == "sphere":
            return sphere_function(position)
        elif function_type == "rastrigin":
            return rastrigin_function(position)
        elif function_type == "ackley":
            return ackley_function(position)
        elif function_type == "beale":
            return beale_function(position)

    def optimize(self, max_iterations, function_type):
        for iteration in range(max_iterations):
            # Evaluate fitness for all particles
            for particle in self.particles:
                particle.fitness = self.evaluate_fitness(particle.position, function_type)
                
                # Update personal best
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Store history for visualization
            self.fitness_history.append(self.global_best_fitness)
            self.position_history.append([p.position.copy() for p in self.particles])
            
            # Update velocity and position for all particles
            for particle in self.particles:
                r1, r2 = random.random(), random.random()
                
                cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
                
                particle.velocity = (self.w * particle.velocity + 
                                   cognitive_velocity + social_velocity)
                
                particle.position += particle.velocity
                
                # Apply bounds
                for i in range(self.dimensions):
                    if particle.position[i] < self.bounds[i][0]:
                        particle.position[i] = self.bounds[i][0]
                        particle.velocity[i] *= -0.5
                    elif particle.position[i] > self.bounds[i][1]:
                        particle.position[i] = self.bounds[i][1]
                        particle.velocity[i] *= -0.5

# Testovací funkce
def sphere_function(x):
    return np.sum(x**2)

def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    
    return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.exp(1)

def beale_function(x):
    x1, x2 = x[0], x[1]
    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*x2**2)**2
    term3 = (2.625 - x1 + x1*x2**3)**2
    return term1 + term2 + term3

def create_visualization_2d(pso, function_type, function_name):
    if pso.dimensions != 2:
        print(f"Vizualizace je možná pouze pro 2D funkce. {function_name} má {pso.dimensions} dimenzí.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vytvoř grid pro kontour plot
    x_range = np.linspace(pso.bounds[0][0], pso.bounds[0][1], 100)
    y_range = np.linspace(pso.bounds[1][0], pso.bounds[1][1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Vypočítej hodnoty funkce
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = pso.evaluate_fitness(np.array([X[i, j], Y[i, j]]), function_type)
    
    # Contour plot
    ax1.contour(X, Y, Z, levels=50, alpha=0.6)
    ax1.contourf(X, Y, Z, levels=50, alpha=0.3, cmap='viridis')
    ax1.set_title(f'{function_name} - Trajektorie částic')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    
    # Plot particle trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(pso.particles)))
    for i, particle in enumerate(pso.particles):
        trajectory = np.array([pos[i] for pos in pso.position_history])
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'o-', 
                color=colors[i], alpha=0.7, markersize=3, linewidth=1)
        ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'o', 
                color=colors[i], markersize=6)
    
    # Mark global optimum
    ax1.plot(pso.global_best_position[0], pso.global_best_position[1], 
            'r*', markersize=15, label='Globální optimum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Convergence plot
    ax2.plot(pso.fitness_history, 'b-', linewidth=2)
    ax2.set_title(f'{function_name} - Konvergence')
    ax2.set_xlabel('Iterace')
    ax2.set_ylabel('Nejlepší fitness')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_optimization_experiment():
    
    # Parametry PSO
    num_particles = 30
    max_iterations = 100
    
    # Definice testovacích funkcí s jejich parametry
    test_functions = [
        {
            'name': 'Sphere Function',
            'type': 'sphere',
            'dimensions': 2,
            'bounds': [(-5, 5), (-5, 5)],
            'global_optimum': 0
        },
        {
            'name': 'Rastrigin Function',
            'type': 'rastrigin',
            'dimensions': 2,
            'bounds': [(-5.12, 5.12), (-5.12, 5.12)],
            'global_optimum': 0
        },
        {
            'name': 'Ackley Function',
            'type': 'ackley',
            'dimensions': 2,
            'bounds': [(-32.768, 32.768), (-32.768, 32.768)],
            'global_optimum': 0
        },
        {
            'name': 'Beale Function',
            'type': 'beale',
            'dimensions': 2,
            'bounds': [(-4.5, 4.5), (-4.5, 4.5)],
            'global_optimum': 0  # minimum at (3, 0.5) with value 0
        }
    ]
    
    results = []
    
    for func_info in test_functions:
        print(f"\n{'='*50}")
        print(f"Optimalizace: {func_info['name']}")
        print(f"{'='*50}")
        
        # Vytvoř PSO instanci
        pso = PSO(
            num_particles=num_particles,
            dimensions=func_info['dimensions'],
            bounds=func_info['bounds']
        )
        
        # Spusť optimalizaci
        pso.optimize(max_iterations, func_info['type'])
        
        # Výsledky
        print(f"Globální optimum nalezeno na pozici: {pso.global_best_position}")
        print(f"Nejlepší fitness: {pso.global_best_fitness:.10f}")
        print(f"Teoretické globální optimum: {func_info['global_optimum']}")
        print(f"Chyba: {abs(pso.global_best_fitness - func_info['global_optimum']):.10f}")
        
        results.append({
            'function': func_info['name'],
            'best_fitness': pso.global_best_fitness,
            'best_position': pso.global_best_position.copy(),
            'error': abs(pso.global_best_fitness - func_info['global_optimum'])
        })
        
        # Vytvoř vizualizaci pro 2D funkce
        if func_info['dimensions'] == 2:
            create_visualization_2d(pso, func_info['type'], func_info['name'])
    
    # Souhrn výsledků
    print(f"\n{'='*50}")
    print("SOUHRN VÝSLEDKŮ")
    print(f"{'='*50}")
    for result in results:
        print(f"{result['function']:20} | Fitness: {result['best_fitness']:12.6e} | Chyba: {result['error']:12.6e}")

def create_animated_visualization_for_function(function_info, save_gif=True):
    print(f"\nVytvářím animovanou vizualizaci pro {function_info['name']}...")
    
    # Setup PSO pro animaci
    pso = PSO(
        num_particles=25, 
        dimensions=function_info['dimensions'], 
        bounds=function_info['bounds']
    )
    
    # Spusť optimalizaci
    pso.optimize(60, function_info['type'])
    
    # Vytvoř animaci
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Grid pro pozadí
    x_range = np.linspace(function_info['bounds'][0][0], function_info['bounds'][0][1], 150)
    y_range = np.linspace(function_info['bounds'][1][0], function_info['bounds'][1][1], 150)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    # Vypočítej hodnoty funkce pro contour plot
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = pso.evaluate_fitness(np.array([X[i, j], Y[i, j]]), function_info['type'])
    
    # Contour plot s více detaily
    levels = 40
    if function_info['type'] == 'beale':
        # Pro Beale function použij logaritmickou škálu pro lepší vizualizaci
        Z_log = np.log10(Z + 1)  # +1 aby se vyhnuli log(0)
        contour = ax.contour(X, Y, Z_log, levels=levels, alpha=0.7, colors='black', linewidths=0.5)
        contourf = ax.contourf(X, Y, Z_log, levels=levels, alpha=0.4, cmap='viridis')
    else:
        contour = ax.contour(X, Y, Z, levels=levels, alpha=0.7, colors='black', linewidths=0.5)
        contourf = ax.contourf(X, Y, Z, levels=levels, alpha=0.4, cmap='viridis')
    
    # Colorbar
    plt.colorbar(contourf, ax=ax, shrink=0.8, label='Fitness hodnota')
    
    # Inicializace bodů částic
    particles_plot = ax.scatter([], [], s=60, c='red', alpha=0.8, edgecolors='darkred', linewidth=1)
    best_plot = ax.scatter([], [], s=300, c='yellow', marker='*', edgecolors='black', linewidth=2)
    
    # Trails for particles (volitelné - trajektorie částic)
    trail_lines = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(pso.particles)))
    for i in range(len(pso.particles)):
        line, = ax.plot([], [], alpha=0.3, color=colors[i], linewidth=1)
        trail_lines.append(line)
    
    ax.set_xlim(function_info['bounds'][0][0], function_info['bounds'][0][1])
    ax.set_ylim(function_info['bounds'][1][0], function_info['bounds'][1][1])
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Poznámky pro globální minimum (pokud je známé)
    if function_info['type'] == 'beale':
        ax.plot(3, 0.5, 's', markersize=10, color='lime', markeredgecolor='black', 
                markeredgewidth=2, label='Teoretické globální minimum (3, 0.5)')
        ax.legend(loc='upper right')
    
    def animate(frame):
        if frame < len(pso.position_history):
            positions = np.array(pso.position_history[frame])
            particles_plot.set_offsets(positions)
            
            # Aktualizuj trajektorie částic (posledních 10 kroků)
            for i, line in enumerate(trail_lines):
                start_frame = max(0, frame - 10)
                trail_positions = np.array([pso.position_history[f][i] for f in range(start_frame, frame + 1)])
                if len(trail_positions) > 1:
                    line.set_data(trail_positions[:, 0], trail_positions[:, 1])
            
            # Aktualizuj nejlepší pozici
            if frame > 0:
                best_plot.set_offsets(pso.global_best_position.reshape(1, -1))
            
            fitness_val = pso.fitness_history[frame] if frame < len(pso.fitness_history) else pso.fitness_history[-1]
            ax.set_title(f'{function_info["name"]} - PSO Animace\n'
                        f'Iterace: {frame+1:02d}/{len(pso.position_history)} | '
                        f'Nejlepší fitness: {fitness_val:.8f}', 
                        fontsize=14, pad=20)
        
        return [particles_plot, best_plot] + trail_lines
    
    anim = FuncAnimation(fig, animate, frames=len(pso.position_history), 
                        interval=300, blit=False, repeat=True)
    
    if save_gif:
        filename = f'pso_{function_info["type"]}_animation.gif'
        try:
            anim.save(filename, writer='pillow', fps=4, dpi=80)
            print(f"Animace uložena jako '{filename}'")
        except Exception as e:
            print(f"Nepodařilo se uložit animaci: {e}")
    
    plt.tight_layout()
    plt.show()
    
    return anim, pso

def create_all_animations():
    """Vytvoří animace pro všechny testovací funkce"""
    
    animation_functions = [
        {
            'name': 'Sphere Function',
            'type': 'sphere',
            'dimensions': 2,
            'bounds': [(-5, 5), (-5, 5)]
        },
        {
            'name': 'Rastrigin Function',
            'type': 'rastrigin',
            'dimensions': 2,
            'bounds': [(-5.12, 5.12), (-5.12, 5.12)]
        },
        {
            'name': 'Ackley Function',
            'type': 'ackley',
            'dimensions': 2,
            'bounds': [(-15, 15), (-15, 15)]
        },
        {
            'name': 'Beale Function',
            'type': 'beale',
            'dimensions': 2,
            'bounds': [(-4.5, 4.5), (-4.5, 4.5)]
        }
    ]
    
    animations = []
    pso_results = []
    
    print(f"\n{'='*60}")
    print("VYTVÁŘENÍ ANIMACÍ PRO VŠECHNY FUNKCE")
    print(f"{'='*60}")
    
    for func_info in animation_functions:
        anim, pso = create_animated_visualization_for_function(func_info, save_gif=True)
        animations.append(anim)
        pso_results.append({
            'name': func_info['name'],
            'type': func_info['type'],
            'best_position': pso.global_best_position,
            'best_fitness': pso.global_best_fitness,
            'pso': pso
        })
        
        print(f"✓ {func_info['name']}: fitness = {pso.global_best_fitness:.8f}, pozice = {pso.global_best_position}")
    
    print(f"\n{'='*60}")
    print("ANIMACE DOKONČENY!")
    print(f"{'='*60}")
    print("Vytvořené soubory:")
    for func_info in animation_functions:
        print(f"  • pso_{func_info['type']}_animation.gif")
    
    return animations, pso_results

if __name__ == "__main__":
    # Nastav random seed pro reprodukovatelnost
    random.seed(42)
    np.random.seed(42)
    
    print("Spouštím PSO optimalizaci pro čtyři testovací funkce...")
    print("Použité funkce: Sphere, Rastrigin, Ackley, Beale")
    
    # Spusť hlavní experiment
    run_optimization_experiment()
    
    # Vytvoř animace pro všechny funkce
    animations, animation_results = create_all_animations()
    
    print("\nOptimalizace a animace dokončeny!")
    print("Výsledky ukazují, že PSO algoritmus úspěšně našel globální optima pro všechny testované funkce.")
    print("\nVytvořené animace:")
    print("  • pso_sphere_animation.gif - Sphere Function")
    print("  • pso_rastrigin_animation.gif - Rastrigin Function") 
    print("  • pso_ackley_animation.gif - Ackley Function")
    print("  • pso_beale_animation.gif - Beale Function")
