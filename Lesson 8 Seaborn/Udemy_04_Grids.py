import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
print(iris)

# Check the unique values
print(iris['species'].unique())

# PAIR PLOT
sns.pairplot(iris)
# plt.show()

# PAIR GRID: It's going to allow us to create a pair plot by our self, with the elements that we want

g = sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
# plt.show()

# FACED GRID: Instead of passing the data base, we choose what is the data, and what column and rows we want.

# DISTPLOT: It only takes one argument.

tips = sns.load_dataset('tips')
g = sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(sns.distplot, 'total_bill')  # To specify on which values we want to make the distplot, pass it as a second
# argument between quotation marks.
plt.show()

# SCATTER PLOT: It takes two arguments.
g = sns.FacetGrid(data=tips, col= 'time', row='smoker')
g.map(plt.scatter,'total_bill', 'tip')  # If the plot it needs a second argument, like the scatter plot. We pass it
# as a third argument between quotation marks
plt.show()

