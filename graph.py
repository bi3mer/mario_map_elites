from os.path import split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

levels_used = 'Mario-1-1'
levels_used = 'All Mario Levels'
# color_bar_label = '# Bad N-Gram Transitions'
color_bar_label = 'Percent Playable'
resolution = 50

cmap = 'Greens'

f = open(os.path.join('data', 'data.csv'))
f.readline()
content = f.readlines()
f.close()

matrix = [[np.nan for _ in range(resolution)] for __ in range(resolution)]
for row in content:
    split_line = row.strip().split(',')
    matrix[int(split_line[1])][int(split_line[0])] = float(split_line[2])
matrix = np.array(matrix)

mask = np.zeros_like(matrix)
for i, row in enumerate(matrix):
    for j, val in enumerate(row):
        if val == np.nan:
            mask[i][j] = 1.0

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.color_palette('viridis')
ax = sns.heatmap(
    matrix, 
    linewidths=.5, 
    square=True, 
    mask=mask,
    cmap=cmap,
    cbar_kws={'label': color_bar_label})
ax.set(xlabel='Linearity', ylabel='Leniency')
ax.set(xticklabels=[], yticklabels=[])
ax.set(title=f"QD Bins with Resolution={resolution} on {levels_used}")
ax.invert_yaxis()
plt.show()