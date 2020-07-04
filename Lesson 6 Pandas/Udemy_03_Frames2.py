import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)
print('')
print('Data Frame')
df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])
print(df)
print('')

print('Ex 1: Conditional selection, higher than 0, (> 0) ')
print(df > 0)
print('')

print('Ex 2: Get the values of the true condition assigning variable ')
booldf = df > 0
print(df[booldf])
print('')

print('Ex 3: get  the values of the True condition without assigning variable')
print(df[df>0])
print('')

print('Ex 4: conditional selection through columns')
print(df['W']>0)
print('')
print('If we want to get only the true values of the condition, not the NaN values, we have to pass the conditional '
      'selection through columns')
print(df[df['W']>0])
print("Notice that we haven't get the C row, where in the W column was negative")
print('')

print('Ex 6: Select in column Z all the values less than 0')
print(df[df['Z']<0])
print('')

print('Ex 7: Call commands in the data frame which we have passed a conditional selection')
resultdf = df[df['W']>0]
print('Data frame which the values are >0 in W column')
print(resultdf)
print('')
print('Column X')
print(resultdf['X'])
print('')
print('Without saving in it a variable')
print(df[df['W']>0]['X'])
print('')
print('Without saving variables and multiple columns')
print(df[df['W']>0][['X', 'Z']])
print('')

print('Ex 8: Multiple conditions')
print('Original Data Frame')
print(df)
print('')

print('And operator')
print(True and True)
print(True and False)
print(False and False)
print(False and  True)
print('')
print('Multiple condition, X and Y higher than 0')
print('')
print(df[(df['W']>0) & (df['Y']>1)])
print('')

print('or operator')
print(True or True)
print(True or False)
print(False or False)
print(False or True)
print('')

print('Multiple condition with or operator |')
print(df[(df['W']>0) | (df['Y']>1)])
print('')

print('Resetting and setting the Idex')
print(df)
print(df.reset_index())
print('')

print('Setting new index')
newind = 'CA NY WY OR CO'.split()
print('We have to set the new index that we want to set in the data frame')
df['State'] = newind
print(df)
print('Once we have it in the data frame, we can set it as the new index')
print(df.set_index('State'))
print("If we don't specify the inplace parametre, we won't make cahanges in the original data frame")
print('')
print(df)
print('')

