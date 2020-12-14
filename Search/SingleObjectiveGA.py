from random import choices, seed
from mario_vglc_grammars.Utility import update_progress
'''
NOTES:
- using single-point crossover
'''


class SingleObjectiveGA:
    __slots__ = [ 'population_generator', 'mutate', 'crossover', 'fitness', 'maximize', 'population_size']

    def __init__(
        self, population_generator, population_size, mutate, crossover, fitness, 
        maximize, rng_seed=None):

        self.population_generator = population_generator
        self.population_size = population_size
        self.crossover = crossover
        self.maximize = maximize
        self.fitness = fitness
        self.mutate = mutate

        if seed != None:
            seed(rng_seed)

    def __add_strand_to_population(self, strand, index, population, population_fitness):
        fitness = self.fitness(strand)
        population_fitness[index] = fitness
        population[index] = strand

    def run(self, epochs):
        '''
        Two arrays are used instead of generating a new one on each epoch to 
        avoid the GC. 
        '''
        population = []
        population_fitness = []

        # Build population. A strand is built via list comprehension where a 
        # a random choice is selected from each of the possible values in 
        # mutation values. This implementation does not allow for continuous
        # values.      
        max_fitness = 0
        population = self.population_generator(self.population_size)
        population_fitness = [self.fitness(strand) for strand in population]
        max_fitness = max(population_fitness)

        old_population = [val for val in population]
        old_population_fitness = [val if self.maximize else abs(max_fitness - val) for val in population_fitness]

        for e in range(epochs):
            update_progress(e / epochs)

            # build new population based on old one
            index = 0
            while index < self.population_size:
                parent_1, parent_2 = choices(old_population, weights=old_population_fitness, k=2)
                strands = self.crossover(parent_1, parent_2)
                for strand in strands:
                    self.__add_strand_to_population(self.mutate(strand), index, population, population_fitness)
                    index += 1

            old_population = [val for val in population]
            old_population_fitness = [val if self.maximize else abs(max_fitness - val) for val in population_fitness]

            if self.maximize:
                max_fitness = max(old_population_fitness)

        update_progress(1)
        return [val for _, val in sorted(zip(population_fitness, population), reverse=self.maximize)]

