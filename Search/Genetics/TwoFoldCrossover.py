from random import randrange

class TwoFoldCrossover:
    __slots__ = ['strand_size']

    def __init__(self, strand_size):
        self.strand_size = strand_size

    def operate(self, parent_1, parent_2):
        '''
        could be improved by having this take in a selector
        '''
        cross_over_point = randrange(0, self.strand_size)

        return [
            parent_1[:cross_over_point] + parent_2[cross_over_point:],
            parent_2[:cross_over_point] + parent_1[cross_over_point:]
        ]
