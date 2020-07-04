import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris

iris = load_iris()

print(iris['DESCR'])

from IPython.display import Image, display

url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
img = Image(url, width=300, height=300)
# How to display images??????

iris = sns.load_dataset('iris')
print(iris)

sns.set_style('whitegrid')

g = sns.PairGrid(data=iris, hue='species', palette='Set2')
g.map_upper(sns.scatterplot)
g.map_lower(sns.scatterplot)

# Bellow for the histograms on the diagonal
d = {}

def func(x, **kwargs):
    ax = plt.gca()

    if not ax in d.keys():
        d[ax] = {'data': [], 'color': []}
    d[ax]['data'].append(x)
    d[ax]['color'].append(kwargs.get('color'))


g.map_diag(func)
for ax, dic in d.items():
    ax.hist(dic['data'], color=dic['color'], histtype='barstacked')

plt.show()

sns.pairplot(iris, hue='species', palette='Dark2')
plt.show()

# creating a density plot

sns.kdeplot(iris[iris['species']=='setosa']['sepal_width'], iris[iris['species']=='setosa']['sepal_length'],
            cmap='magma', shade=True, shade_lowest=False)
plt.show()

setosa = iris[iris.species == 'setosa']
print(setosa)

setosa = iris.loc[iris.species == 'setosa']

sns.kdeplot(setosa.sepal_length, setosa.sepal_width, cmap='plasma', shade=True, shade_lowest=False)
plt.show()


X = iris.drop('species', axis=1)

y = iris['species']

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3, random_state=101)

model = SVC()

model.fit(X_train, y_train)

predict = model.predict(X_test)

print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=2)

grid.fit(X_train, y_train)

print(grid.best_params_)

print(grid.best_estimator_)

grid_pred = grid.predict(X_test)

print(confusion_matrix(y_test, grid_pred))
print('\n')
print(classification_report(y_test, grid_pred))

