import numpy as np
import pandas as pd
from numpy.random import randn
# Index Levels
outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
inside = [1, 2, 3, 1, 2, 3]
hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
print('')

print('Multiply levels')
print(hier_index)
print('')

print('Ex 1: Data frame with multiply levels')
df = pd.DataFrame(randn(6,2), hier_index, ['A', 'B'])
print(df)
print('')

print('Index for Multiply levels')
print(df.loc['G1'])
print('')
print(df.loc['G1'].loc[1])
print('It will return the value of row one as a series')
print('')

print('Ex 2: Index names')
print(df.index.names)
df.index.names = ['Groups', 'Num']
print(df)
print('')

print('Ex 3: Getting values')
print(df.loc['G1'].loc[1]['B'])
print('')

print('Ex 4: Getting values, second method')
print(df.xs(1, level='Num'))
