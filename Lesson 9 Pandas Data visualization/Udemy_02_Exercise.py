import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df3 = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\07-Pandas-Built-in-Data-Viz\df3")
print(df3)
print(df3.info())

# Ex 1 Scatter Plot
df3.plot.scatter(x='a', y='b', figsize=(12, 3), c= 'red', s=50, edgecolor='black')
plt.show()

# Ex 2 Histogram plot
df3['a'].plot.hist(edgecolor='black')
plt.show()

# Ex 3 Polish the plot
plt.style.use('ggplot')  # to use different types of style by passing the name as a string:
# https://matplotlib.org/gallery.html#style_sheets
df3['a'].plot.hist(color='red', edgecolor='white', bins=25, alpha=0.7)
plt.show()

# Ex 3 Box Plot
df3[['a', 'b']].plot.box(color='r', whiskerprops={'linestyle':'--', 'linewidth': '1.5'})  # If we  don't want all the
# columns plot, but more than once, we have to pass a list
plt.show()

# KDE Plot
df3['d'].plot.kde()
plt.show()

# KDE Plot, changing the linestyle and line width
df3['d'].plot.kde(linestyle='--', linewidth=2)  # If its a line 2d, we haven't to specify the
# whiskerprops property
plt.show()

# Area Plots
df3[:30].plot.area(alpha=0.8)
plt.show()

# Area Plot: setting the legend outside

df3[0:30].plot.area(alpha=0.4)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # In other IDE we should define the figure as f = plt.figure()
plt.title('Plot 1')
plt.show()


# f = plt.figure()
# df3[0:30].plot.area(alpha=0.4,ax=f.gca())
# plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.title('Plot 2')
# plt.show()