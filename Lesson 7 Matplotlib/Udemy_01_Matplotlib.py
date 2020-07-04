import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 11)
y = x ** 2
print(x)
print(y)

# FUNCTIONAL METHOD
plt.plot(x, y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
#plt.show()

# Multi Plots
plt.subplot(1, 2, 1)  # specifying that the plot has 1 roe two columns and we're going to specify under this the plot
# nº1
plt.plot(x, y, 'r')  # Specifying the plot number 1
plt.subplot(1, 2, 2)
plt.plot(y, x, 'b')  # Specifying the plot number 2
#plt.show()  # Showing the plot

#  Multi plots ex 2: 4 plots
plt.subplot(2, 2, 1)
plt.plot(x, y, 'r:')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title('Plot Number 1')
plt.subplot(2, 2, 2)
plt.plot(y, x, 'b-.')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title('Plot Number 2')
plt.subplot(2, 2, 3)
plt.plot(x - 20, y, 'g-')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title('Plot Number 3')
plt.subplot(2, 2, 4)  # We can edit the plot under the subplot (we can change the order)
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title('Plot Number 4')
plt.plot(y, (x + 20), 'm--')
#plt.show()

# Object oriented method
# Created figured objects and then just call method off of this
fig = plt.figure()
print(fig)
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # It's going to ask 4 arguments, a left bottom width and height
# basically the percent of that blank canvas we want to go ahead.
axes.plot(x, y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')
#plt.show()  # We can only show the plot once we've specified the axes

#       Multi plot using Object Method: Plot inside a plot
fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes1.plot(x, y)
axes2 = fig.add_axes([0.5, 0.1, 0.4, 0.3])
axes3 = fig.add_axes([0.1, 0.6, 0.4, 0.3])
#plt.show()

#       Two plots
fig = plt.figure()
axes4 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
axes.plot(x, y)
axes5 = fig.add_axes([0.55, 0.1, 0.4, 0.8])
#plt.show()

#       Plot Exercise
fig = plt.figure()
axes6 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes7 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
axes6.plot(x, y)
axes7.plot(y, x)
axes6.set_title('Bigger Plot')
axes7.set_title('Smaller Plot')
#plt.show()  # To see all the changes we have to put it at the end of the code, if not we'll only see util where it is
