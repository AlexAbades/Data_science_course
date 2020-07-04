import numpy as np
import pandas as pd

df = pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': [444, 555, 666, 444],
    'col3': ['abc', 'def', 'ghi', 'xyz']
})
print(df.head())  # Head function, by default shows the 5th firsts rows, specify different number
print('')

print('Ex 1: Find what unique value are in the column')
print(df['col2'].unique())
print('')

print('Ex 2: Show how many unique values are in a column')
print('1st Method:')
print(len(df['col2'].unique()))
print('')
print('2nd Method:')
print(df['col2'].nunique())
print('')

print('Ex 3: Unique values for all de data series')
print(type(df.nunique()))
print('')
print(df.nunique())
print('')

print('Ex 4: Counting the times that appear a value, by column values')
print(df['col2'].value_counts())
print('')

print('Ex 5: Conditional selection, remember')
print(df[(df['col1'] > 2) & (df['col2'] == 444)])
print('')

print('Ex 6 Apply function')


def times2(x):
    return x * 2


# As we know, we can select a column and use a built in function on it, like the sum function
# If we want to apply our own functions to the Data frame, we can use the apply method
# For example, if we ant to calculate the length of the strings of every element in col3
# we'll have to use a foor loop, or... we can use the apply method
print('Example of built in function in a column')
print(df['col1'])
print('')
print('The sum of the col1 is:')
print(df['col1'].sum())
print('')
print('If we want tu use our times 2 function we have to do it:')
print(df['col1'].apply(times2))
print('')
print('So, if we want to calculate the length of string that we have in all our data base:')
print(df['col3'].apply(len))
print('')
print('We can combine tha apply function with the lambda expressions')
print(df['col2'].apply(lambda x : x*2))
print('')

print('Ex 7: Drop Columns')
print(df.drop('col1', axis=1))
print('')

print('Ex 8:See the columns and the indexes')
print(df.columns)
print(df.index)
print('')

print('Ex 9: Sort values by value ')
print(df.sort_values('col2'))
# The index stays attached to the row
print('')

print('Ex 10: Find no Values')
print(df.isnull())
print('It returns booleans values')
print('')

data = {
    'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
    'B': ['one', 'one', 'two', 'two', 'one', 'one'],
    'C': ['x', 'y', 'x', 'y', 'x', 'y'],
    'D': [1, 3, 2, 5, 4, 1]
}
df1 = pd.DataFrame(data)
print(df1)
print('')
print('Pivot tables')
print(df1.pivot_table(values='D', index=['A', 'B'], columns='C'))
