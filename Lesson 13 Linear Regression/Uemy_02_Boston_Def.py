import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.datasets import load_boston

boston = load_boston()

print(boston['DESCR'])
print('')
print('')

print(boston)
print('')

df = pd.DataFrame(data=boston.data, columns=boston['feature_names'])
print('X Values')
print(df)
print('')

print('Y Values')
y = pd.DataFrame(data=boston['target'], columns=['House price'])
print(y)
print('')

print('Info')
print(df.info())

df['House price']=y['House price']

print(df)
sns.pairplot(df)
plt.show()

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True,cmap= 'coolwarm', ax=ax)
plt.show()