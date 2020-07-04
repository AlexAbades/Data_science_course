# Plotly is an interactive visualization library Cufflinks connects plotly with Pandas, in that way we're going to be
# able to create interactive visualization directly with our data frame

import pandas as pd
import numpy as np
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly.offline.plot as ply


init_notebook_mode(connected=True)  # This it's going to connect the java script to our notebook because plotly
# essentially connect pandas and python to an interactive javascript library.

cf.go_offline()  # Allows to use cufflinks offline
df = pd.DataFrame(np.random.randn(100, 4), columns="A B C D".split())
print(df)

df2 = pd.DataFrame({'Category':['A', 'B', 'C'], 'Values':[32, 43, 50]})
print(df2)

# To use the interactive Plotly, you should use this option! The Key here, it' that we have to specify the parameter
# "asFigure" the "filename" parameter it's going to save your file with the name you specify
plot(df.iplot(asFigure=True), filename='Probe_1.html')

# Scatter Plot

plot(df.iplot(kind='scatter', x='A', y='B', mode='markers', size=20, asFigure=True), filename='Scatter_plot.html')
# By default Plotly it's going to make a line trying to connect all the points, to get a really scatter plot,
# you have to specify the "mode" argument to put a maker in each value!

# BAR PLOT
plot(df2.iplot(kind='bar', x='Category', y='Values', asFigure=True), filename='Bar_plot.html')

# CALLING AGREGATE FUNCTIONS IN PLOTLY

plot(df.sum().iplot(kind='bar', asFigure=True), filename='Agregate_function.html')

# BOX PLOT

plot(df.iplot(kind='box', asFigure=True), filename='Box_plot.html')

# 3D SURFACES

df3 = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 20, 10], 'z':[500, 400, 300, 200, 100]})
print(df3)

# 1
plot(df3.iplot(kind='surface', asFigure=True), filename='Surface_plot.html')

# 2
df4 = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 20, 10], 'z':[5, 4, 3, 2, 1]})
plot(df3.iplot(kind='surface', colorscale='rdylbu', asFigure=True), filename='Surface_2_plot.html')
# The color Scale it's to specify the color scale that we want into our plot

# HISTOGRAM PLOT

plot(df.iplot(kind='hist', asFigure=True), filename='Hist_plot.html')

# We can make a histogram for only a column

plot(df['A'].iplot(kind='hist', asFigure=True), filename='Hist_A_plot.html')

# SPREAD PLOT (diferencia)

plot(df[['A', 'B']].iplot(kind='spread', asFigure=True), filename='Spread_plot.html')

# BUBBLE PLOT; Size of the bubbles based in values of other columns

plot(df.iplot(kind='bubble', x='A', y='B', size='C', asFigure=True), filename='Bubble_plot.html')

# SCATTER MATRIX
# Very similar to the Seaborn pair plot. You've to be sure that all your column

plot(df.scatter_matrix(asFigure=True), filename='Scatter_matrix_plot.html')

# To have more information :https://github.com/santosjorge/cufflinks
# https://plotly.com/python/v3/figure-labels/
# Financial: https://github.com/santosjorge/cufflinks/blob/master/cufflinks/ta.py