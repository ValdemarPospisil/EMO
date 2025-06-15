import random

class KnapsackProblem:

    def __init__(self, items: list[tuple[int, int]], capacity: int):
        self.items = items
        self.capacity = capacity
        self.num_items = len(items)

    def fitness(self, solution: list[int]) -> int:
        total_weight = 0
        total_value = 0
        for i in range(self.num_items):
            if solution[i] == 1:
                total_weight += self.items[i][0]
                total_value += self.items[i][1]

        # Penalizace za překročení kapacity
        if total_weight > self.capacity:
            return 0
        else:
            return total_value

    def get_random_solution(self) -> list[int]:
        
        return [random.randint(0, 1) for _ in range(self.num_items)]

    def __str__(self):
        return f"KnapsackProblem(items={self.num_items}, capacity={self.capacity})"
