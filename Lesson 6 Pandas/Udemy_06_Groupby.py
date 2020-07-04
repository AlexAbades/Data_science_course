import numpy as np
import pandas as pd

data = {
    'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
    'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
    'Sales': [200, 120, 340, 124, 243, 350]
}
df = pd.DataFrame(data)
print(df)
byComp = df.groupby('Company')
print('')

print('Ex 1: Mean')
print(byComp.mean())
print('')

print('Ex 2: Sum')
print(byComp.sum())
print('')

print('Ex 3: Standard deviation')
print(byComp.std())
print('')

print('Ex 4: Indexing by groupby')
print(byComp.sum().loc['FB'])
print('')

print('Ex 5: Indexing, all in one')
print(df.groupby('Company').sum().loc['FB'])
print('')

print('Ex 6: Count function')
print(df.groupby('Company').count())
print('')

print('Ex 7: Max function')
print(df.groupby('Company').max())
print('')

print('Ex 7: Min function')
print(df.groupby('Company').min())
print('')

print('Ex 8: Describe and transpose function')
print(df.groupby('Company').describe().transpose())