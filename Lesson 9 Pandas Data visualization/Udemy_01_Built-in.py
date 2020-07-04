import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# All of the following commands, or different ways to make plots, are directly compiled by Pandas, You are using the
# pandas core. It's for a quick visualization. The difference it's how we are calling to the plot!

df1 = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\07-Pandas-Built-in-Data-Viz\df1",
                  index_col=0)
print(df1)

df2 = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\07-Pandas-Built-in-Data-Viz\df2")
print(df2)

# Histogram plot with pandas
df1['A'].hist(bins=30)
plt.show()

# Making a plot with the data frame itself
sns.set_style('darkgrid')
df1['A'].plot(kind='hist')
plt.show()

# 2nd Method
df1['A'].plot.hist(bins=30)
plt.show()

# Area Plot
df2.plot.area(alpha=0.4)
plt.show()

# Bar Plot
df2.plot.bar()
plt.show()
# It takes the index value as a category, you should be sure thar your index is has categorical values. In these case
# it works because our data frame it's small, only 9 rows

# Bar plot with staked argument

df2.plot.bar(stacked=True)
plt.show()

# Histogram plot

df2.plot.hist(bins=30)
plt.show()

# Line Plot
# we have to specify the x and y
print(df1.index)
df1_index = df1.index
df1.plot.line(y='B', lw= 0.5)  # Check the probe at the bottom of the script!
plt.show()
# We should get the same result if we won't specify the x column

# Scatter Plot with colored by values
df1.plot.scatter(x='A', y='B', c='C', cmap='coolwarm')  # With the color parameter,
# we can say that the plot it's going to be colored by the values of a column. In this case we could say that we have
# a 3D plot.
plt.show()

# Scatter Plot with the point sized by values of a column
df1.plot.scatter(x='A', y='B', s=df1['C']*100, edgecolors='black')  # In the size parameter, we have to pass a data
# frame column in self; We can multiply that column by some sort of factor
plt.show()

# Box Plots
df2.plot.box()
plt.show()

# Hexagonal Plot

df = pd.DataFrame(np.random.randn(1000,2), columns=['A', 'B'])
print(df)
df.plot.hexbin(x='A', y='B', gridsize=25, cmap='coolwarm')  # If we want to set the size, we can pass the "gridsize"
# parameter. As the hex plot that we saw in the seaborn lesson, the hexagons goes darker as more pints are inside of
# them.
plt.show()

# KDE Kernel density estimation
df2['a'].plot.kde()
plt.show()

# You can also call it with:
df2['a'].plot.density()
plt.show()

# And we also can call this KDE plot to all the data frame.
df2.plot.kde()
plt.show()















print(' Probe \n ')
# Looks like python has been updated so that it will use the index by default. That means you can remove the x axis
# portion of the code in this case.
# Getting values from a Unnamed column
df11 = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\07-Pandas-Built-in-Data-Viz\df1")
print(df11)
print(df11['Unnamed: 0'])
df11.plot.line(x='Unnamed: 0', y='B')
plt.title('df11')
plt.show()
print(df11.columns.str.match('Unnamed'))
# That returns a list of boolean values with length of the nº columns.
# [ True False False False False]

print('')
# Rename a Unnamed column, You can get it with 'Unnamed: 0'.
df11.rename(columns={'Unnamed: 0': 'Dates'}, inplace=True)
print(df11)

# So now we can reference to that column with the name Dates