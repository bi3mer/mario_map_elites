from mario_vglc_grammars.Generation.Constrained import generate_from_start_to_end
from random import normalvariate
from math import ceil, floor

class NGramMutate:
    __slots__ = [
        'standard_deviation', 'mutation_values', 'mutation_rate', 
        'max_length', 'gram']

    def __init__(self, mutation_rate, gram, max_length, standard_deviation = 3):
        '''
        max length should be longer than the original strand length.
        '''
        self.standard_deviation = standard_deviation
        self.mutation_rate = mutation_rate
        self.max_length = max_length
        self.gram = gram

    def mutate(self, strand):
        # we use slightly less than half for a smaller path
        path = None
        while path == None:
            mid_normal_point = len(strand) / 3
            points = [
                normalvariate(mid_normal_point, self.standard_deviation),
                normalvariate(mid_normal_point, self.standard_deviation)
            ]

            s = max(self.gram.n + 1, floor(min(points)))
            e = max(self.gram.n + 1, ceil(max(points)))

            path = generate_from_start_to_end(self.gram, strand[:s], strand[e:], e-s)

        return path[:self.max_length]
        