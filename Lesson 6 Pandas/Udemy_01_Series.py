import numpy as np
import pandas as pd

labels = ['a', 'b', 'c']
my_data = [10, 20, 30]
arr = np.array(my_data)
d = {
    'a': 10,
    'b': 20,
    'c': 30}

print('Normal Series')
s = pd.Series(my_data)
print(s)
print('')

print('Series with modified index')
s1 = pd.Series(my_data, labels)
print(s1)
print(s1['a'])
print('')

print('Series with an array')
s2 = pd.Series(arr, labels)
print(s2)
print('')
print('')

print('Series with dictionary')
s3 = pd.Series(d)
print(s3)
print('')

print('Series with strings data')
s4 = pd.Series(data = labels)
print(s4)
print('')

print('Series with functions')
s5 = pd.Series(data=[sum, print,len])
print(s5)
print('')

print('Indexing series')
s6 = pd.Series([1, 2, 3, 4], ['USA', 'Germany', 'USSR', 'Japan'])
print(s6)
s7 = pd.Series([1, 2, 5, 4], ['USA', 'Germany', 'Italy', 'Japan'])
print(s7)
print('')
print(s6)
print('')
s8 = pd.Series(labels)
print(s8)
print(s8[1])
print('')
print(s6+s7)
print(s6/s7)