from mario_vglc_grammars.Fitness.Playability import naive_percent_playable
from mario_vglc_grammars.Fitness import bad_transitions, linearity, leniency, max_linearity, percent_playable
from mario_vglc_grammars.Utility import columns_into_level_string
from mario_vglc_grammars.Grammar import NGram, UniGram
from mario_vglc_grammars.IO import *
from Search import MapElites
from Search.Genetics import *

from collections import deque
from itertools import repeat
from csv import writer
import sys
import os

# =================== Configuration ===================
grammar_size = 3
min_path_length = 3
strand_size = 25
max_length = 30
start_population_size = 500
iterations = 10000000
# iterations = 100000
# iterations = 1000
resolution = 50
mutation_rate = 0.02
seed = 0

percent_performance_switch = 1/1000
minimize_performance = False

# =================== Set Up Data Storage ===================
if not os.path.isdir('data'):
    os.mkdir('data')

level_path = os.path.join('data', 'levels')
if not os.path.isdir(level_path):
    os.mkdir(level_path)
elif len(sys.argv) == 2 and sys.argv[1] == '-o':
    data_file = os.path.join('data', 'data.csv')
    if os.path.exists(data_file):
        os.remove(data_file)
    for filename in os.listdir(level_path):
        os.remove(os.path.join(level_path, filename))
else:
    print(f'ERROR: Will not overwrite data. Please delete directory {level_path}')
    sys.exit(1)

# =================== Grammar Training ===================
levels = [get_single_super_mario_bros('mario-1-1.txt')]
levels = get_super_mario_bros()
unigram = UniGram()
gram = NGram(grammar_size)

for level in levels:
    gram.add_sequence(level)
    unigram.add_sequence(level)

mutation_values = [unigram.keys for _ in repeat(None, strand_size)]

# =================== Map-Elites ===================
feature_names = ['linearity', 'leniency']
feature_descriptors = [linearity, leniency]
feature_dimensions = [
    [0, max_linearity(strand_size, 16)], 
    [0, strand_size]] 

population_generator = PopulationGenerator(mutation_values, strand_size).generate
mutator = Mutate(mutation_values, mutation_rate, strand_size).mutate
crossover = TwoFoldCrossover(strand_size).operate

start_sequence = levels[0][:strand_size + 1]
population_generator = NGramPopulationGenerator(gram, start_sequence, strand_size).generate
mutator = NGramMutate(mutation_rate, gram, max_length).mutate
crossover = NGramCrossover(gram, 0, max_length).operate

me = MapElites(
    start_population_size, 
    feature_descriptors, 
    feature_dimensions, 
    resolution,
    naive_percent_playable,
    percent_playable,
    percent_performance_switch, 
    minimize_performance, 
    population_generator, 
    mutator, 
    crossover, 
    rng_seed=seed
)

me.run(iterations)

# =================== Save ===================
f = open(os.path.join('data', 'data.csv'), 'w')
writer = writer(f)
writer.writerow(feature_names + ['performance'])

for i, key in enumerate(me.bins.keys()):
    writer.writerow(list(key) + [me.bins[key][0]])

    level_file = open(os.path.join(level_path, f'{i}.txt'), 'w')
    level_file.write(columns_into_level_string(me.bins[key][1]))
    level_file.close()

f.close()
