import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
print(tips)

sns.set_style('darkgrid')  # You can fix a lot of styles, 'white', 'dark', 'darkgrid', 'ticks', 'whitegrid'
sns.countplot(x='sex', data=tips)
sns.despine()  # It's to quite the spines, or to understand it better, the axes. By default, top and tight are True,
# but you quit more passing the parameters.
plt.show()

# MATPLOTLIB functions to size the plot
plt.figure(figsize=(12, 3))
sns.countplot(x='sex', data=tips)
plt.show()

# CONTEXT METHOD
sns.set_context('poster')  # It has predefined contexts, so if we know that we are going to print it after,
# we can specify it.
# The "font_scale" parameter it's going to multiply the size of the real.
sns.countplot(x='sex', data=tips)
plt.show()