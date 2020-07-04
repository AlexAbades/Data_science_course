import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')
print(titanic)
print(titanic.info())

# Exercise 1: Jointplot

sns.set_style('whitegrid')
sns.jointplot(x='fare', y='age', data=titanic, marginal_kws=dict(bins=20))  # Bins number of columns
plt.show()

# Exercise 2: Distplot

sns.set_style('darkgrid')  # If we want to set a style grid, it has to be before we call the plot
sns.distplot(titanic['fare'], kde=False, color='r', bins=30)
plt.show()

# Exercise 3: Boxplot

sns.boxplot(x='class', y='age', data=titanic, palette='rainbow')
plt.show()

# Exercise 4: Swarm Plot
sns.swarmplot(x='class', y='age', data=titanic, palette='Set2')
plt.show()

# Ex 5: count plot
sns.countplot(x='sex', data=titanic)
plt.show()

# Ex 6: Heat Map
# We have to make a matroix before we can plot this type of plot.
ti_heat = titanic.corr()
print(ti_heat)
sns.heatmap(ti_heat, cmap='coolwarm')
plt.title('titanic.corr()')
plt.show()

# Ex 7 two distplot

g = sns.FacetGrid(data=titanic, col='sex')
g.map(sns.distplot, 'age', kde=False)
g.set(xlim=(0, 80), ylim=(0, 120), xticks=np.arange(0, 90, 10))
plt.show()


g = sns.FacetGrid(data=titanic, col='sex')
g.map(sns.hist, 'age')
g.set(xlim=(0, 80), ylim=(0, 120), xticks=np.arange(0, 90, 10))
plt.show()
