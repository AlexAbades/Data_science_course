import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,100)
y = x*2
z = x**2
print(x)

# Ex 1
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x, y)
ax.set_title('Title')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim([0, 100])
ax.set_ylim([0, 200])
#plt.show()

# Ex 2

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.set_xlim([0, 1])
ax2 = fig.add_axes([0.2, 0.5, .2, .2])
ax2.set_xlim([0, 1])
plt.xticks(np.arange(0, 1.1, 0.2))  # xticks to set the divisions per axe, in this case "X". We pass the list of the
# divisions that we want.
print(np.arange(0, 1.1, 0.2))  # Remember, the max value it's not included.
plt.yticks(np.arange(0, 1.1, 0.2))
#plt.show()


# Ex 3

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = fig.add_axes([0.2, 0.5, 0.2, .2])
ax1.set_title('Exercise 3')
ax2.set_title('S')
ax1.plot(x, y)
ax2.plot(x, y)
#plt.show()

# Ex 4
fig = plt.figure()
ax1 = fig.add_axes([0.15, 0.1, 0.8, .8])
ax2 = fig.add_axes([.25, .5, .4, .3])
ax1.plot(x, z)
ax2.plot(x, y)
ax2.set_xlim([20, 22])
ax2.set_ylim([30, 50])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Ex 4')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('zoom')
# plt.show()

# Ex 5
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(x, y, ls='--', c='b', lw=5)
ax[1].plot(x, z, ls='-', c='r', lw=5)
ax[0].set_title('Ex 5')
ax[1].set_title('Red')
# plt.show()

# Ex 6
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 2))
axes[0].plot(x, y, c='b', ls='-', lw=4)
axes[1].plot(x, z, c='r', ls='--', lw=4)
plt.show()