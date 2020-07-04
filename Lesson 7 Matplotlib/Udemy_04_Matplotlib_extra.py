import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 21)
print(x)
y = x ** 2
print(y)

# Look out! When we are using the functional method subplot, we haven't to specify anything, it's subplot, without a s
plt.subplot(1, 2, 1)
plt.plot(x, y, 'g')
plt.subplot(1, 2, 2)
plt.plot(y, x, 'r')
#plt.show()

# Ex 1

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
axes[0].plot(x, y, 'r--')
axes[1].plot(y, x, 'g', marker='*', mfc='black', mec='black', ms=10)
plt.show(block=True)

#plt.show()

# Ex 2

fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
axes1.plot(x, y, 'b')
axes2.plot(y, x, 'r')
plt.show(block=True)

#plt.show()

# Ex 3

fig, ax = plt.subplots()
ax.plot(x, x, c='Red', ls=':')
ax.plot(x, x+10, c='Blue', ls='-.')
ax.plot(x, x+20, c='magenta', ls= '--', alpha=0.2)
plt.show(block=True)

#plt.show()

# Ex 4

fig, ax = plt.subplots()
ax.plot(x, x+15, color='b', linestyle='--', marker='1')
ax.plot(x, x+14, color='purple', linestyle='-', marker='s', ms=5, mfc='yellow', mew=2, mec='green')
ax.plot(x, x+13, color='purple', linestyle='-', marker='o', ms=5, mfc='red')
ax.plot(x, x+12, color='purple', lw= 0.75, linestyle='-', marker='o', ms=4)
ax.plot(x, x+11, color='purple', lw=0.5, linestyle='-', marker='o', ms=2)
ax.plot(x, x+10, c='b', ls= '--', marker='o', ms=4)
ax.plot(x, x+9, c='b', marker='+' )
line, = ax.plot(x, x+8, color='black', linestyle='-.')
line.set_dashes([5, 10, 15, 5])  # set_dashes([line length, space length...])
ax.plot(x, x+7, color='green', linestyle=':')
ax.plot(x, x+6, c='green', ls='-.', lw= 3)
ax.plot(x, x+5, c='green', lw=2, ls='-')
ax.plot(x, x+4, c='r', lw=2, alpha=0.4)
ax.plot(x, x+3, c='red', lw=2)
ax.plot(x, x+2, c='red', linewidth=1.5)
ax.plot(x, x+1, c='red', lw=1)
ax.plot(x, x, c='red', linewidth=0.5)
plt.show(block=True)

#plt.show()

# Ex 5

fig, ax = plt.subplots()
ax.plot(x, y, 'b', label='Pau')
ax.plot(x, x**2.4, 'g', label='Alex')
ax.set_title('Good at games')
ax.set_xlabel('Played hours')
ax.set_ylabel('Accuracy')
ax.set_xlim([4, 17.5])
ax.set_ylim([0, 1000])
plt.legend()
plt.show(block=True)

#plt.show()

# Ex 6
plt.scatter(y, x)
plt.show(block=True)


# Ex 7
# Bars graphic
import random

# We can use, from random import sample. Then we won't have to use random.sample, only sample

data=random.sample(range(1, 1000), 100)
print(data)
plt.hist(data)
plt.show()



