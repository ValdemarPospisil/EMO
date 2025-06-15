import operator
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio
from deap import base, creator, tools, gp, algorithms
from tqdm import tqdm

class VisualizerCallback:
    def __init__(self, problem, save_interval=5):
        self.problem = problem
        self.save_interval = save_interval
        self.image_files = []

    def __call__(self, population, generation, hall_of_fame):
        if generation % self.save_interval == 0 or generation == self.problem.generations:
            best_ind_of_gen = tools.selBest(population, 1)[0]
            
            ant_sim = AntSimulator(self.problem.trail_map, self.problem.max_moves)
            program = self.problem.toolbox.compile(expr=best_ind_of_gen)
            
            while ant_sim.moves < ant_sim.max_moves:
                program(ant_sim)
                
            filename = self.visualize_ant_path(ant_sim, generation)
            self.image_files.append(filename)

    def visualize_ant_path(self, ant, generation, save_dir="frames"):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = ListedColormap(['#FFFFFF', '#aaddff', '#FFD700'])
        display_map = np.zeros_like(ant.trail_map, dtype=int)
        display_map[ant.trail_map == 1] = 2
        path_y, path_x = zip(*ant.path)
        display_map[path_y, path_x] = 1
        ax.imshow(display_map, cmap=cmap, interpolation='nearest')
        ax.plot(path_x, path_y, 'b-', alpha=0.5, linewidth=1.5)
        ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='Konec')
        ax.set_title(f"Generace: {generation} | Snědeno: {ant.eaten_food}/{int(np.sum(ant.trail_map))} | Kroky: {ant.moves}")
        ax.set_xticks([]); ax.set_yticks([])
        filename = os.path.join(save_dir, f"frame_{generation:04d}.png")
        plt.savefig(filename, dpi=120)
        plt.close(fig)
        return filename

    def create_evolution_gif(self, output_path="evolution.gif"):
        if not self.image_files: return
        images = [imageio.imread(f) for f in sorted(self.image_files) if os.path.exists(f)]
        if images:
            imageio.mimsave(output_path, images, duration=0.4, loop=0)
            print(f"GIF animace uložena jako: {output_path}")

class AntSimulator:
    def __init__(self, trail_map, max_moves):
        self.trail_map = trail_map
        self.max_moves = max_moves
        self.height, self.width = self.trail_map.shape
        self.reset()
    def reset(self):
        self.y, self.x = 0, 0; self.direction = 1
        self.moves = 0; self.eaten_food = 0
        self.path = [(self.y, self.x)]; self.world = self.trail_map.copy()
        if self.world[self.y, self.x] == 1: self.eaten_food += 1; self.world[self.y, self.x] = 0
    def _get_front_coords(self):
        if self.direction == 0: return (self.y - 1) % self.height, self.x
        elif self.direction == 1: return self.y, (self.x + 1) % self.width
        elif self.direction == 2: return (self.y + 1) % self.height, self.x
        else: return self.y, (self.x - 1) % self.width
    def is_food_ahead(self):
        front_y, front_x = self._get_front_coords()
        return self.world[front_y, front_x] == 1
    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1; self.y, self.x = self._get_front_coords()
            self.path.append((self.y, self.x))
            if self.world[self.y, self.x] == 1: self.eaten_food += 1; self.world[self.y, self.x] = 0
    def turn_left(self):
        if self.moves < self.max_moves: self.moves += 1; self.direction = (self.direction - 1) % 4
    def turn_right(self):
        if self.moves < self.max_moves: self.moves += 1; self.direction = (self.direction + 1) % 4

class AntProblem:
    def __init__(self, map_data, max_moves=600, generations=50):
        self.trail_map = np.array(map_data)
        self.max_moves = max_moves
        self.generations = generations
        self._setup_deap()

    def _setup_deap(self):
        class AntAction: pass
        self.pset = gp.PrimitiveSetTyped("MAIN", [], AntAction)
        self.pset.addPrimitive(lambda o1, o2: lambda ant: (o1(ant), o2(ant)), [AntAction, AntAction], AntAction, name="prog2")
        def if_food_ahead(out1, out2): return lambda ant: out1(ant) if ant.is_food_ahead() else out2(ant)
        self.pset.addPrimitive(if_food_ahead, [AntAction, AntAction], AntAction)
        self.pset.addTerminal(lambda ant: ant.move_forward(), AntAction, "move_fwd")
        self.pset.addTerminal(lambda ant: ant.turn_left(), AntAction, "turn_L")
        self.pset.addTerminal(lambda ant: ant.turn_right(), AntAction, "turn_R")

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=6)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evaluate_ant)
        self.toolbox.register("select", tools.selTournament, tournsize=7)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def _evaluate_ant(self, individual):
        try:
            program = self.toolbox.compile(expr=individual)
            ant = AntSimulator(self.trail_map, self.max_moves)
            while ant.moves < ant.max_moves:
                program(ant)
            return ant.eaten_food,
        except Exception:
            return 0,

    def run_evolution(self, pop_size=300, cxpb=0.7, mutpb=0.2):
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean); stats.register("max", np.max)
        
        visualizer = VisualizerCallback(self)
        
        self.toolbox.register("evaluate_generation", visualizer)

        print("Spouštím evoluci...")
        
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)
        
        visualizer(pop, 0, hof)

        for gen in tqdm(range(1, self.generations + 1), desc="Evoluce"):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = algorithms.varAnd(offspring, self.toolbox, cxpb, mutpb)
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)
            
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
            
            visualizer(pop, gen, hof)

        print("\n--- Evoluce dokončena ---")
        visualizer.create_evolution_gif()
        
        print("\nNejlepší nalezené řešení:")
        best_ind = hof[0]
        print(f"  Fitness (snědeno jídla): {best_ind.fitness.values[0]}")
        print(f"  Velikost programu (počet uzlů): {len(best_ind)}")


if __name__ == "__main__":
    santa_fe_map_data = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]

    random.seed(42)
    np.random.seed(42)
    
    problem = AntProblem(santa_fe_map_data)
    problem.run_evolution()
