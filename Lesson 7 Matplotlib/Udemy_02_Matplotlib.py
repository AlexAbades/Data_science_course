import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 11)
y = x ** 2
print(x)
print(y)
w, z = [5, 6]
print(w)
print(z)

# Creating multi plots with object oriented

# Ex 1

fig, axes = plt.subplots(nrows=1, ncols=2)
print(axes)
# Tuple unpacking for "fig,axes". Axes (object) it's just an array of matplotlib axes, which are ones that we
# manually created with the fig.add_axes and it's iterable, also, it will create an array or a matrix depending on
# how many rows and columns we specify: nrows= 1, ncols=2 it will create an array with two elements; nrows = 4,
# ncols= 4, it sill create a matrix of 4 rows and 4 elements. As many arrays as . The difference between the
# plt.figure() and plt.subplots() it's that the second one makes the fig.add_axes([]) automatically based on how many
# rows and columns you want. We can iterate through that axes object.
for current_ax in axes:
    current_ax.plot(x, y)
#plt.show()

# plt.tight_layout()  To solve the overlapping problems

# Ex 2
# As the axes object it's iterable, we can index it!
fig, axes1 = plt.subplots(1, 2)
axes1[0].plot(x, y)
axes1[0].set_title('First Plot')
axes1[1].plot(y, x)
axes1[1].set_title('Second Plot')
#plt.show()

fig, axes1 = plt.subplots(2, 2)
axes1[0, 0].plot(x, y)
axes1[0, 0].set_title('1st')
axes1[0, 1].plot(x, y)
axes1[0, 1].set_title('2nd')
axes1[1, 0].plot(x, y)
axes1[1, 0].set_title('3th')
axes1[1, 1].plot(x, y)
axes1[1, 1].set_title('4th')
#plt.show()

# Figure size (in inches), aspect ratio and DPI (dots per inch or pixels per inch)
fig = plt.figure(figsize=(3, 2))
ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
plt.tight_layout()
#plt.show()
# Possibles solutions to the error: C:\Program Files\JetBrains\PyCharm
# 2019.3.4\plugins\python\helpers\pycharm_matplotlib_backend\backend_interagg.py:64: UserWarning: This figure
# includes Axes that are not compatible with tight_layout, so results might be incorrect. self.figure.tight_layout()
# File > Settings > Tools >  Python > show plots in tool window
# plt.savefig('pic.png',bbox_inches='tight')

fig, axes2 = plt.subplots(nrows=2, ncols=1, figsize=(8, 2), dpi=100)
axes2[0].plot(x, y)
plt.tight_layout()
#plt.show()

# Saving the matplotlib
fig, axes2 = plt.subplots(nrows=2, ncols=1, figsize=(8, 2), dpi=100)
axes2[0].plot(x, y)
plt.tight_layout()
fig.savefig('my_picture.png')  # We only have to specify the extension that we want, we can specify the dpi here.

# Legends

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x, x**2, label='Banana')  # 3th argument: label, it's for the legend
ax.plot(x, x**3, label='Strawberries')
ax.legend(loc=0)  # You have to specify that you want the legend, but after you've specified the label. You can
# choose the location of the legend with the loc= argument. It has predefined values (0-10) take a look to:
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend
# We can also specify the location by passing a tuple which indicates the left bottom corner. loc= (0.1, 0.1)
#plt.show()
