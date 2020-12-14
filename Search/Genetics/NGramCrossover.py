from mario_vglc_grammars.Generation.Constrained import generate_from_start_to_end
from random import normalvariate
from math import floor

class NGramCrossover:
    __slots__ = ['gram', 'min_length', 'max_length', 'standard_deviation']

    def __init__(self, gram, min_length, max_length, standard_deviation=3):
        self.standard_deviation = standard_deviation
        self.max_length = max_length
        self.min_length = min_length
        self.gram = gram

    def operate(self, parent_1, parent_2):
        '''
        could be improved by having this take in a selector
        '''
        strand_size = min(len(parent_1), len(parent_2))
        paths = None
        mid_normal_point = strand_size / 3
        while paths == None:
            cross_over_point = floor(normalvariate(mid_normal_point, self.standard_deviation))
            cross_over_point = max(self.gram.n + 1, cross_over_point)

            p_1 = generate_from_start_to_end(
                self.gram, parent_1[:cross_over_point], 
                parent_2[cross_over_point:], 
                self.min_length)

            if p_1 == None:
                continue

            p_2 = generate_from_start_to_end(
                self.gram, 
                parent_2[:cross_over_point], 
                parent_1[cross_over_point:], 
                self.min_length)
            
            if p_2 == None:
                continue

            paths = [p_1, p_2]

        paths[0] = paths[0][:self.max_length]
        paths[1] = paths[1][:self.max_length]
        
        return paths
