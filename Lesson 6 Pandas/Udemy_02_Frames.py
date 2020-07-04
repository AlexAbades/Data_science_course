import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)  #To set to all the same random numbers.
print('Ex 1')
df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])
print(df)
print('')

print('Ex 2')
print('Index, column W')
print(df['W'])
print('')

print('Ex 3')
print('Type')
print(type(df['W']))
print('')

print('Ex 4')
print('An other method to extract columns is')
print(df.W)
# We should avoid this type of notation,  because it could overwrite one of the predefined methods, then panda will
# get confused
print('')

print('Ex 5')
print(df[['W', 'Z']])
print('')

print('Ex 6')
print('Making new columns')
df['New'] = df['W'] + df['Y']
print(df)
print('')

print('Ex 7')
print('Making new columns and new data base')
df['New1'] = [4, 5, 6, 7, 8]
print(df)
print('')

print('Ex 8')
print('Drop a column')
print(df.drop(['New', 'New1'], axis=1, inplace=True))
print(df)
print('')

print('Ex 9')
print('Drop rows')
print(df.drop('E'))
print('')

print('Ex 10')
print('Frame shape')
print(df.shape)
print('5 rows, 4 columns. Index 0 is rows, index 1 is columns')
print('')

print('Ex 11')
print('Selecting rows: A')
print(df.loc['A'])
print('')

print('Ex 12')
print('2nd method: C')
print(df.iloc[2])  # We can get the row from the index, even the row has a label.
print('')

print('Ex 13')
print('Selecting subsets of rows & columns')
print(df.loc['A','W'])
print('')

print('Ex 14')
print('2nd method, select a more than once')
print(df.loc[['A','B'], ['W', 'Y']])

# Print only two columns
print(df[['W', 'X']])