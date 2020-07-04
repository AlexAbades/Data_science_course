import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plotly.offline import plot
import numpy as np
import plotly.graph_objs as go
import cufflinks as cf
import plotly

cf.go_offline()

bank_stocks = pd.read_pickle(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\10-Data-Capstone"
                             r"-Projects\all_banks")

tickers = 'BAC C GS JPM MS WFC'.split()
ticks = []
for element in tickers:
    ticks.append(element + ' Returns')
print(ticks)

# Creating a dataframe with only the close price
BAC = bank_stocks.xs(key='Close', axis=1, level=1)
print(BAC)

# Creating a matrix of the correlation of close price
BAC_corr = BAC.corr()

# Creating a Heat map
print(BAC_corr)
sns.heatmap(BAC_corr, cmap='coolwarm', annot=True)
plt.show()

# Creating a cluster map
sns.clustermap(BAC_corr, annot=True, cmap='coolwarm')
plt.show()

# Creating heatmap plotly
plot(BAC_corr.iplot(kind='heatmap', asFigure=True, colorscale='rdylbu'), filename='heatmap.html')

# Creating candle plot form BAC 2015-2016
bac = bank_stocks['BAC'][['Open', 'High', 'Low', 'Close']].loc['2015-01-01':'2016-01-01']
plot(bac.iplot(kind='candle', asFigure=True), filename='Cadnleplot.html')

# Creating Simple Moving Average from Morgan Stanley 2015

MS = bank_stocks['MS']['Close'].loc['2015-01-01':'2016-01-01']
plot(MS.ta_plot(study='sma', asFigure=True, periods=[13, 21, 55]), filename='MorganStanley_Moving_Average.html')

# Bollinger band Plot for bank of America; the standard deviation of the stock price
plot(bac['Close'].ta_plot(study='boll', asFigure=True), filename='BAC Bollinger.html')
# Creating candle plot with average

fig = go.Figure(data=[
    go.Candlestick(
        x=bac.index,
        open=bac['Open'],
        high=bac['High'],
        low=bac['Low'],
        close=['Close']
    )])
plot(fig, filename='Candle_Plot_Average.html')