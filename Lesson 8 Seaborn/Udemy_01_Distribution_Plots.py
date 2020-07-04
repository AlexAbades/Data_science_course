# http://seaborn.pydata.org/
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Seaborn actually comes with a few data base that we can actually load.
tips = sns.load_dataset('tips')
print(tips)

# dist plot: allows us to  show the distribution os a univariate set of observations "One variable distribution"
# We only have to pass a column
sns.distplot(tips['total_bill'], kde=False, bins=100)
plt.show()

# It's basically an histogram graph with a kde "Kernel density estimation" --> non parametric model. It's used to
# study study. https://www.youtube.com/watch?v=lpSIWtNQsVY To put it or not, kde argument with a boolean value We can
# make the histogram graphic with more bins (contenedores o columnas). With the "bins" argument. If the number of
# bins is to high, we'll get a weird scenario, getting one bin for tip

# To run only the kde plot we can use the
sns.kdeplot(tips['total_bill'])
plt.show()

sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
plt.show()

# Join plot allows us to match up to "distplot" for bivariate (two variables) and unvariate data. Meaning we can
# essentially combine two different distribution plots. Histogram plots. To make that plot, we have to pass three
# parameters, x, y, and data. "x" & "y" are the columns that we want to compare, and the data is variable name of the
# data frame, as a variable we have to put it without quotation marks, we haven't to pass the data and the column:
# tips['total_bill], only 'total_bill' between quotation marks. By default is scatter, but we can change the inside
# plot with the "kind" argument. Hexagon distribution, if the hexagon has more pints inside it gets darker
# kind, 'reg' like a scatter plot but with a regression line, and pearson parameter; 'kde', allows us to get a two
# dimensional kde, which essentially shows you the density where these points much up the most.

sns.pairplot(tips, hue='sex', palette='coolwarm')
plt.show()
# It's essentially going to plot pairwise relationships across an entire data frame, at least for the numerical columns.
# But essentially what going to do is plot a jointplot for every single possible combination of the numerical columns
# in this data frame.
# The important thing about pairplot is that you can pass a hue argument "hue". You are going to pass the column name of
# categorical column, (where there ara categories, for example, in this data base the sex column)
# As a third argument, we can pass the pallet, which allows us to change th colour for pairplot.

# Create a rug plot
sns.rugplot(tips['total_bill'])
plt.xlim(0, 60)
plt.ylim(0, 1)
plt.show()
# It draw a dash mark for every single point along the distribution line. It's really similar to the histogram plot.


# Don't worry about understanding this code!
# It's just for the diagram below
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create dataset
dataset = np.random.randn(25)

# Create another rugplot
sns.rugplot(dataset)

# Set up the x-axis for the plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2

# 100 equally spaced points from x_min to x_max
x_axis = np.linspace(x_min, x_max, 100)

# Set up the bandwidth, for info on this:
url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'

bandwidth = ((4 * dataset.std() ** 5) / (3 * len(dataset))) ** .2

# Create an empty kernel list
kernel_list = []

# Plot each basis function
for data_point in dataset:
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
    kernel_list.append(kernel)

    # Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * .4
    plt.plot(x_axis, kernel, color='grey', alpha=0.5)

plt.ylim(0, 1)
plt.show()

# The normal distribution is build in top of each this blue dashes

# To get the kde plot we can sum these basis functions.

# Plot the sum of the basis function
sum_of_kde = np.sum(kernel_list,axis=0)

# Plot figure
fig = plt.plot(x_axis,sum_of_kde,color='indianred')

# Add the initial rugplot
sns.rugplot(dataset,c = 'indianred')

# Get rid of y-tick marks
plt.yticks([])

# Set title
plt.suptitle("Sum of the Basis Functions")
plt.show()

