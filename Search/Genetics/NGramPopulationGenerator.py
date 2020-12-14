from itertools import repeat

from mario_vglc_grammars.Generation.Unconstrained import generate

class NGramPopulationGenerator:
    __slots__ = ['strand_size', 'start_sequence', 'n_gram']

    def __init__(self, n_gram, start_sequence, strand_size):
        self.start_sequence = start_sequence
        self.strand_size = strand_size
        self.n_gram = n_gram

    def generate(self, n):
        return [generate(self.n_gram, self.start_sequence, self.strand_size) for _ in repeat(None, n)]
