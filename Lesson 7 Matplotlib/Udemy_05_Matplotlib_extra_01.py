import matplotlib.pyplot as plt
import numpy as np
from random import sample

data = sample(range(1, 1000), 100)
print(data)
plt.hist(data)
plt.show()

data1= [np.random.normal(0, std, 100) for std in range(1, 4)]
print(data1)
plt.boxplot(data1, vert=True, patch_artist=True)
plt.show()
# http://www.matplotlib.org - The project web page for matplotlib. https://github.com/matplotlib/matplotlib - The
# source code for matplotlib. http://matplotlib.org/gallery.html - A large gallery showcaseing various types of plots
# matplotlib can create. Highly recommended! http://www.loria.fr/~rougier/teaching/matplotlib - A good matplotlib
# tutorial. http://scipy-lectures.github.io/matplotlib/matplotlib.html - Another good matplotlib reference.
