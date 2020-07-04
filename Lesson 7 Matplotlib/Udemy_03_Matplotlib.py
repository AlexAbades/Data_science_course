import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 11)
y = x**2
print(x, y)
print(len(x), len(y))

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x, y, color='#FF8C00', lw=0.5, alpha=1, ls='steps', marker='o', ms=10, mfc='yellow',mew=3, mec='green')
# We can pass common colour names as a string in the parameter "color" or "c". Or we can specify RGV hex codes,
# also between quotation marks. For the line width, we can set it with "linewidth" or "lw", by default it's 1. For
# the transparency of the line we can use the alpha argument, which allows you how transparent the line is. We can
# change the line style with the "linestyle" argument or the abbreviation "ls" , we have to pass it as a string. We
# can find a lot of styles: https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
# .set_linestyle We can put markers at the points that we have with the "marker" argument, also as a string, '+',
# 'o', '*', 's', ',', '.', '1', '2', '3', '4'... We also can modify the size of the marker with the "markersize" or
# abbreviate it with "ms". As a number. Marker face color, "markerfacecolor" argument or the abbreviation "mfc". Also
# between strings. We can change the marker edge width with the argument "markeredgewidth" or the abbreviation "mew".
# To change the marker edge color, we can specify it with the "markeredgecolor" or the abbreviation "mec". Also
# between quotation marks.
plt.tight_layout()
#plt.show()

# Control over axis appearance

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x, y, lw=2, ls='--', c='r')
# To set the axes limits, we can specify it with the .set_xlim & set_ylim, passing a list of the lower and upper values
ax.set_xlim([0, 1])
ax.set_ylim([0, 2])
#plt.show()

