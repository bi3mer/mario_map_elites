from random import choice, random

class Mutate:
    __slots__ = ['mutation_values', 'mutation_rate', 'strand_size']

    def __init__(self, mutation_values, mutation_rate, strand_size):
        self.mutation_values = mutation_values
        self.mutation_rate = mutation_rate
        self.strand_size = strand_size

    def mutate(self, strand):
        return [
            strand[i] if random() > self.mutation_rate else choice(self.mutation_values[i])
            for i in range(self.strand_size)]        
