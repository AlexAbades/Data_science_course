import numpy as no
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.datasets import load_boston

boston = load_boston()
print(boston['DESCR'])

print('boston')
print(boston)
print(type(boston))


boston_df = boston.data
print('boston_df')
print(boston_df)
print(type(boston_df))

column_names = boston['feature_names']
print(column_names)

X = pd.DataFrame(data= boston_df, columns=column_names)
print(X)

target = boston['target']
y = pd.DataFrame(target)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print('X_train type:')
print(type(X_train))
print('X_test type: ')
print(type(X_test))
print('y_train type:')
print(type(y_train))
print('y_test type:')
print(type(y_test))

lm = LinearRegression()


print(lm.fit(X_train, y_train))

print(lm.intercept_)

print(lm.coef_)
print(X_train.columns)

cfd = pd.DataFrame(data=lm.coef_, columns=X_train.columns, index=['Coeff'])
print(cfd)

predictions = lm.predict(X_test)

print(predictions)
print(type(predictions))

print(y_test)
arr= y_test.to_numpy()
print(arr)
sns.scatterplot(x=[arr], y=[predictions])
