from itertools import repeat
from random import choice

class PopulationGenerator:
    __slots__ = ['mutation_values', 'strand_size']

    def __init__(self, mutation_values, strand_size):
        self.mutation_values = mutation_values
        self.strand_size = strand_size

    def generate(self, n):
        return [[choice(self.mutation_values[i]) for i in range(self.strand_size)] for _ in repeat(None, n)]
