from random import seed,  sample
from math import floor

from mario_vglc_grammars.Utility import update_progress

class MapElites:
    '''
    This is the more basic form of map-elites without resolution switching,
    parallel execution, etc.

    NOTE: change the mutation to instead use a uni-gram rather than the array
    of arrays. 
    '''
    __slots__ = [
        'feature_descriptors', 'feature_dimensions',  'resolution', 'performance', 
        'minimize_performance', 'bins', 'keys', 'start_population_size',
        'population_generator', 'mutator', 'crossover']

    def __init__(
        self, start_population_size, feature_descriptors, feature_dimensions, 
        resolution, performance, minimize_performance, population_generator, 
        mutator, crossover, rng_seed=None):

        self.minimize_performance = minimize_performance
        self.feature_descriptors = feature_descriptors
        self.feature_dimensions = feature_dimensions
        self.resolution = 100 / resolution # view __add_to_bins comments
        self.performance = performance
        self.start_population_size = start_population_size
        self.population_generator = population_generator
        self.mutator = mutator
        self.crossover = crossover
        self.bins = None
        self.keys = None

        if seed != None:
            seed(rng_seed)

    def run(self, iterations):
        self.bins = {} 
        self.keys = set()
        
        for strand in self.population_generator(self.start_population_size):
            self.__add_to_bins(strand)

        for i in range(iterations - self.start_population_size):
            parent_1 = self.bins[sample(self.keys, 1)[0]][1]
            parent_2 = self.bins[sample(self.keys, 1)[0]][1]

            for strand in self.crossover(parent_1, parent_2):
                self.__add_to_bins(self.mutator(strand))

            update_progress(i / iterations)

        update_progress(1)

    def __add_to_bins(self, strand):
        '''
        Resolution is the number of bins for each feature. Meaning if we have 2
        features and a resolution of 2, we we will have a 2x2 matrix. We have to
        get take scores and map them to the indexes of the matrix. We get this 
        by first dividing 100 by the resolution which will be used to get an index
        for a mapping of a minimum of 0 and a max of 100. We are given a min and
        max for each dimension of the user. We take the given score and convert it
        from their mappings to a min of 0 and 100. We then use that and divide the
        result by the 100/resolution to get a float. When we floor it, we get a valid
        index given a valid minimum and maximum from the user.

        Added extra functionality to allow for additional fitness if the main fitness
        is found to be equal to the current best fitness
        '''
        fitness = self.performance(strand)
        feature_vector = [score(strand) for score in self.feature_descriptors]
        
        for i in range(len(self.feature_dimensions)):
            minimum, maximum = self.feature_dimensions[i]
            score = feature_vector[i]
            score_in_range = (score - minimum) * 100 / maximum 
            feature_vector[i] = floor(score_in_range / self.resolution)

        feature_vector = tuple(feature_vector)
        if feature_vector not in self.bins:
            self.keys.add(feature_vector)
            self.bins[feature_vector] = [fitness, strand]
        else:
            current_fitness_score = self.bins[feature_vector][0]

            if self.minimize_performance and fitness > current_fitness_score:
                self.bins[feature_vector][0] = fitness
                self.bins[feature_vector][1] = strand
            elif fitness <   current_fitness_score:
                self.bins[feature_vector][0] = fitness
                self.bins[feature_vector][1] = strand

