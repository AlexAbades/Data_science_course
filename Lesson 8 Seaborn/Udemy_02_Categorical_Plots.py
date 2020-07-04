import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
print(tips)
print(type(tips['sex'][5]))
print('')
print('')
# Seeing the distribution of a categorical column in reference to another categorical column or a numerical column

# BAR PLOT
# It's just a general plot that allows you to aggregate the categorical data based in some function, by default the mean
# You can think of these as visualization of a group by action.

# In the x argument we're going to pass a categorical column, and in the y argument we're going to pass a numerical
# column estimator parameter it's is a function, an aggregate function, a statistical function to estimate within
# each categorical bin.
sns.barplot(x='sex', y='total_bill', data=tips)
plt.show()

# Cannot cast array data from dtype('int64') to dtype('int32') according to the rule 'safe' Don'tknow why is giving
# an error: TypeError: Cannot cast array data from dtype('int64') to dtype('int32') according to the rule 'safe'

#######################################################
sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)
plt.show()
########################################################


# Count plot: Is the same as the barplot except the estimator is explicitly counting the number of occurrences
sns.countplot(x='sex', data=tips)
plt.show()
# It like sat pandas.count() of this sex column for gender. Counts the number of males and females


# Box plots and violin plots used to shown the distribution of a categorical data. Box plot is used to show the
# distribution og a quantitative data in a way that hopefully facilities comparision between variables

sns.boxplot(x='day', y='total_bill', data=tips)
plt.show()


# This plot shows the core tiles of the data set while the whiskers extend to show the rest of the distribution,
# except of the out point, these are determined to be outliers. https://en.wikipedia.org/wiki/Box_plot Divided in
# quarters, 25% the lowest whiskers to the bottom of the box. lower to the box to the middle.... If you turned 90º,
# you can represent the plot as a normal gaussian distribution

sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')
plt.show()

# x for category values or columns, and the y to numerical values
sns.violinplot(x='day', y='total_bill', data=tips)
plt.show()

# The violin plot unlike the box plot, allows us to actually plot all the components that correspond to actual data
# point. And it's also essentially showing the kernel density estimation of the underlying distribution. If we can of
# split this in half the distribution points on in side. If you cut one of the violin figure it in half in the y
# axes and rote it 90º, it would be like a kendal distribution.

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex')
plt.show()

# It also allows us to make the hue argument. That it allow us to split up the box, or violin plots even further by
# another categorical o divide our data and compare it columns

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True)
plt.show()
# The split argument, it allows us to show in one side of the violin plot, one value of the hue (the second
# categorical split) and in the other side the other value.
# Violin plots, for people eho have seen more of this plots.
# To CEO or other business men, the box plot.

sns.stripplot(x='day', y='total_bill', data=tips, jitter=True, hue='sex', dodge=True)
plt.show()
# The jitter argument makes more distance between the point, we also can pass the hue and the split argument.
# The split argument it also can be write it as dodge


# SWARM PLOT
# Its' the combination og the scatter plot and the violin plot.
sns.swarmplot(x='day', y='total_bill', data=tips, color='black')
sns.violinplot(x='day', y='total_bill', data=tips)
plt.show()

# We can combine two different plots in the same figure, we can change the color with the color argument to whow it
# better

# CATPLOT PLOT
# A generic call to plots, you have to specify the kind of the plot with the kind argument.
sns.catplot(x='day', y='total_bill', data=tips, kind='violin')
plt.show()