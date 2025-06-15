import pygame
import neat
import os
import random
import math

# --- Globální konstanty a proměnné ---
WIN_WIDTH = 500
WIN_HEIGHT = 800
GENERATION = 0

# Načtení obrázků
try:
    BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
                 pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
                 pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
    PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
    BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
    BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
except pygame.error:
    print("Chyba: Ujisti se, že máš obrázky 'bird1-3.png', 'pipe.png', 'base.png', 'bg.png' ve složce 'imgs/'")
    quit()

pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 50)


# --- Třídy pro herní objekty ---

class Bird:
    """Reprezentuje jednoho ptáka ve hře."""
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]
        self.distance_traveled = 0  # Pro sledování ujeté vzdálenosti

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """Vypočítá a aplikuje fyziku pohybu ptáka (gravitace)."""
        self.tick_count += 1
        self.distance_traveled += 1  # Zvýší ujetou vzdálenost
        
        # Výpočet posunu (d = v0*t + 1/2*a*t^2)
        displacement = self.vel * self.tick_count + 1.5 * self.tick_count**2
        
        # Omezení maximální rychlosti pádu
        if displacement >= 16:
            displacement = 16
        # Zjemnění skoku
        if displacement < 0:
            displacement -= 2
        
        self.y = self.y + displacement

        # Naklánění ptáka
        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        """Vykreslí ptáka s animací křídel a nakláněním."""
        self.img_count += 1
        # Animace mávání křídel
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0
        
        # Zabráníme mávání při pádu
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        # Otočení obrázku
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    """Reprezentuje překážku (dvojici trubek)."""
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(150, 450)  # Rozšířen rozsah pro větší variabilitu
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        """Detekce kolize ptáka s trubkou."""
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False


class Base:
    """Reprezentuje posouvající se zem."""
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


# --- Hlavní funkce a smyčka ---

def draw_window(win, birds, pipes, base, score, gen, best_fitness):
    """Vykreslí všechny prvky na obrazovku."""
    win.blit(BG_IMG, (0,0))

    for pipe in pipes:
        pipe.draw(win)
    
    base.draw(win)

    for bird in birds:
        bird.draw(win)

    # Vykreslení textů
    score_label = STAT_FONT.render("Skóre: " + str(score), 1, (255,255,255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    
    gen_label = STAT_FONT.render("Generace: " + str(gen), 1, (255,255,255))
    win.blit(gen_label, (10, 10))
    
    alive_label = STAT_FONT.render("Naživu: " + str(len(birds)), 1, (255,255,255))
    win.blit(alive_label, (10, 50))
    
    # Zobrazení nejlepší fitness
    fitness_label = STAT_FONT.render("Nejlepší: " + str(round(best_fitness, 1)), 1, (255,255,255))
    win.blit(fitness_label, (10, 90))

    pygame.display.update()


def calculate_fitness_bonus(bird, pipe, score):
    """Vypočítá bonus fitness na základě pozice ptáka vůči trubce."""
    # Bonus za létání ve středu mezery
    gap_center = pipe.height + pipe.GAP / 2
    distance_from_center = abs(bird.y - gap_center)
    center_bonus = max(0, 1.0 - distance_from_center / (pipe.GAP / 2))
    
    # Progresivní bonus za delší přežití
    survival_bonus = score * 0.1
    
    return center_bonus * 0.05 + survival_bonus


def normalize_input(value, min_val, max_val):
    """Normalizace vstupu do rozsahu -1 až 1."""
    return 2 * (value - min_val) / (max_val - min_val) - 1


def eval_genomes(genomes, config):
    """
    Fitness funkce volaná NEATem. Spustí hru pro celou generaci.
    """
    global GENERATION
    GENERATION += 1

    nets = []
    ge = []
    birds = []

    # Vytvoření ptáků a jejich neuronových sítí
    for _, genome in genomes:
        genome.fitness = 0  # Vždy začít s fitness 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)

    base = Base(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0
    best_fitness = 0

    run = True
    while run and len(birds) > 0:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # Určení, na kterou trubku se zaměřit (první vpředu)
        pipe_ind = 0
        if len(birds) > 0 and len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

        # Pohyb a rozhodování ptáků
        for x, bird in enumerate(birds[:]):  # Create a copy for safe iteration
            if x >= len(ge) or x >= len(nets):  # Safety check
                continue
                
            bird.move()
            
            # Základní fitness za přežití (sníženo)
            ge[x].fitness += 0.01
            
            # Bonus fitness za dobrou pozici
            if len(pipes) > pipe_ind:
                fitness_bonus = calculate_fitness_bonus(bird, pipes[pipe_ind], score)
                ge[x].fitness += fitness_bonus

            # Rozšířené vstupy pro neuronovou síť
            current_pipe = pipes[pipe_ind] if len(pipes) > pipe_ind else pipes[0]
            next_pipe = pipes[pipe_ind + 1] if len(pipes) > pipe_ind + 1 else None
            
            # Normalizované vstupy
            bird_y_norm = normalize_input(bird.y, 0, WIN_HEIGHT)
            bird_vel_norm = normalize_input(bird.vel, -16, 16)
            
            # Vzdálenost k aktuální trubce
            horizontal_dist = current_pipe.x - bird.x
            horizontal_dist_norm = normalize_input(horizontal_dist, -100, 400)
            
            # Pozice vzhledem k aktuální trubce
            pipe_top_dist = bird.y - current_pipe.top
            pipe_bottom_dist = current_pipe.bottom - bird.y
            pipe_center = current_pipe.height + current_pipe.GAP / 2
            center_dist = bird.y - pipe_center
            
            pipe_top_norm = normalize_input(pipe_top_dist, -200, 400)
            pipe_bottom_norm = normalize_input(pipe_bottom_dist, -200, 400)
            center_dist_norm = normalize_input(center_dist, -300, 300)
            
            # Vstupy pro následující trubku (pokud existuje)
            next_pipe_inputs = [0, 0, 0] if next_pipe is None else [
                normalize_input(next_pipe.x - bird.x, 200, 800),
                normalize_input(bird.y - (next_pipe.height + next_pipe.GAP / 2), -300, 300),
                normalize_input(next_pipe.height, 150, 450)
            ]

            # Kompletní vstup pro neuronovou síť (9 vstupů)
            inputs = [
                bird_y_norm,           # Pozice ptáka
                bird_vel_norm,         # Rychlost ptáka
                horizontal_dist_norm,  # Horizontální vzdálenost k trubce
                pipe_top_norm,         # Vzdálenost k horní trubce
                pipe_bottom_norm,      # Vzdálenost ke spodní trubce
                center_dist_norm,      # Vzdálenost ke středu mezery
                next_pipe_inputs[0],   # Další trubka - horizontální vzdálenost
                next_pipe_inputs[1],   # Další trubka - vertikální pozice
                next_pipe_inputs[2]    # Další trubka - výška
            ]

            # Rozhodnutí o skoku
            output = nets[x].activate(inputs)
            if output[0] > 0.5:
                bird.jump()
                # Malá penalizace za časté skákání
                ge[x].fitness -= 0.002

        base.move()

        add_pipe = False
        rem = []
        for pipe in pipes:
            # Kontrola kolizí - iterujeme přes kopii seznamu
            for x, bird in enumerate(birds[:]):
                if x >= len(ge):  # Safety check
                    continue
                    
                if pipe.collide(bird):
                    # Penalizace za kolizi
                    ge[x].fitness -= 5
                    # Malý bonus za délku přežití
                    ge[x].fitness += bird.distance_traveled * 0.001
                    
                    # Odstranění ptáka a jeho sítě
                    if x < len(nets):
                        nets.pop(x)
                    if x < len(ge):
                        ge.pop(x)
                    if x < len(birds):
                        birds.pop(x)

            # Kontrola, jestli pták proletěl trubkou
            if len(birds) > 0:
                if not pipe.passed and pipe.x < birds[0].x:
                    pipe.passed = True
                    add_pipe = True

            # Odstranění trubek mimo obrazovku
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        # Přidání nové trubky a snížená odměna za průlet
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 1.5
            pipe_distance = random.randint(550, 650)
            pipes.append(Pipe(pipe_distance))

        # Odstranění trubek
        for r in rem:
            pipes.remove(r)

        # Kontrola, zda ptáci nevyletěli z obrazovky
        for x, bird in enumerate(birds[:]):  
            if x >= len(ge):
                continue
                
            if bird.y + bird.img.get_height() - 10 >= 730 or bird.y < -50:
                ge[x].fitness -= 3
                ge[x].fitness += bird.distance_traveled * 0.001
                
                if x < len(nets):
                    nets.pop(x)
                if x < len(ge):
                    ge.pop(x)
                if x < len(birds):
                    birds.pop(x)

        if ge:
            current_best = max(g.fitness for g in ge)
            best_fitness = max(best_fitness, current_best)

        draw_window(win, birds, pipes, base, score, GENERATION, best_fitness)
        
        if score > 100:
            for g in ge:
                g.fitness += score * 0.5
            break

def run(config_path):
    """Spustí celý proces neuroevoluce."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(eval_genomes, 100)
    print('\nNejlepší genom:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
