import numpy as np
import pandas as pd

d = {
    'A':[1, 2, np.nan],
    'B':[5, np.nan, np.nan],
    'C':[1, 2, 3]
}
df = pd.DataFrame(d)
print(df)
print('')

print('Ex 1: dropna method rows')
print(df.dropna())
print('')

print('Ex 2: dropna method into columns')
print(df.dropna(axis=1))
print('')

print('Ex 3: Tesh argument, specifies how many non-Na values, we will need to pass the row')
print(df.dropna(thresh=2))
print('')

print('Ex 4: Filling values')
print(df.fillna(value='Empty'))
print('')

print('Ex 5: Filling values with the mean')
print(df['A'].fillna(value=df['A'].mean()))
