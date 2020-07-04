import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check the versions
print(np.version.version)
print(sns.__version__)

# Basic regression plot
tips = sns.load_dataset('tips')
sns.lmplot(x='total_bill', y='tip', data=tips)
plt.show()

# Regression plot with hue

sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex')
plt.show()

# Regression plot with hue and markers

sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex',
           markers=['o', 'v'])  # We have to pass a list of markers, because we ara making a hue, so we'll need a
# marker for each category, so remember, MARKERS, in plural !
plt.show()

# Adding arguments to modify the plot.

# Seaborn works using the matplotlib library, so when we want to customize the plot, we'll have to get inside the
# plot we want to modify of matplotlib, so we will pass a dictionary with the arguments to modify the plot as arguments

sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', markers=['o', 'v'],
           scatter_kws={
               's': 100
           })
plt.show()

# Dividing the plot in columns instead of colors

sns.lmplot(x='total_bill', y='tip', data=tips, col='sex')
plt.show()

# Dividing in rows and columns

sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', row='time')
plt.show()

# We can use the hue argument with the the col and row parameter, but it'll probably show to much information

sns.lmplot(x='total_bill', y='tip', data=tips, col='day', hue='sex', aspect=0.6, height=8, palette='coolwarm')
plt.show()
# Aspect ratio is the parameter which one we can modify the high and the width of the plot. The size, is the size.
# Palette it's the parameter to change the "color" of our plot. We can find more in matplotlib colormap
# https://matplotlib.org/tutorials/colors/colormaps.html



