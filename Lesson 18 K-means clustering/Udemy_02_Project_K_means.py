import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

univ = pd.read_csv(
    r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\17-K-Means-Clustering\College_Data",
    index_col=0)
print(univ)

print(univ.info())

print(univ.describe())

sns.set_style('whitegrid')
sns.scatterplot(x='Room.Board', y='Grad.Rate', data=univ, hue='Private')
plt.show()

sns.lmplot(x='Room.Board', y='Grad.Rate', data=univ, hue='Private', fit_reg=False, palette='coolwarm', size=6, aspect=1)
plt.show()

sns.scatterplot(x='Outstate', y='F.Undergrad', hue='Private', data=univ)
plt.show()

pal = dict(No='salmon', Yes='dodgerblue')
g = sns.FacetGrid(univ, hue='Private', palette=pal, height=8)
g = (g.map(plt.hist, 'Outstate', bins=20, alpha=.5)
     .add_legend())
plt.show()

g = sns.FacetGrid(univ, hue='Private', palette=pal, height=8)
g = (g.map(plt.hist, 'Grad.Rate', bins=20, alpha=.5)
     .add_legend())
plt.show()

print(univ[univ['Grad.Rate'] > 100])

univ['Grad.Rate']['Cazenovia College']=100

g = sns.FacetGrid(univ, hue='Private', palette=pal, height=8)
g =(g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.5)
    .add_legend())
plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

data = univ.drop('Private', axis=1)

kmeans.fit(data)

print(len(kmeans.labels_))
print('\n')
print(kmeans.cluster_centers_)

univ['Cluster'] = univ['Private'].apply(lambda x: 1 if x == 'Yes' else 0)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(univ['Cluster'], kmeans.labels_))
print('\n')
print(classification_report(univ['Cluster'], kmeans.labels_))