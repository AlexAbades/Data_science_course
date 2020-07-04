import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import plotly
import cufflinks as cf
import plotly.graph_objs as go

cf.go_offline()

tickers = 'BAC C GS JPM MS WFC'.split()
new_tickers = []
for tick in tickers:
    new_tickers.append(tick + ' Returns')
print(new_tickers)

bank_stocks = pd.read_pickle(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\10-Data-Capstone"
                             r"-Projects\all_banks")
returns = pd.DataFrame()
for new, old in zip(new_tickers, tickers):
    returns[new] = bank_stocks[old]['Close'].pct_change()
print(returns.head())

# Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns.
# You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that
# day?
print('Min')
print(returns.idxmin())
print('')
print('Max')
print(returns.idxmax())
print('')

# Standard deviation

# Standard deviation is the statistical measure of market volatility, measuring how widely prices are dispersed from
# the average price. If prices trade in a narrow trading range, the standard deviation will return a low value that
# indicates low volatility. Conversely, if prices swing wildly up and down, then standard deviation returns a high
# value that indicates high volatility.

print('Standard deviation')
print(returns.std())
print('')

# Selecting only the values of 2015 and calculating the standard deviation
print('Standard deviation in 2015')
print(returns.loc['2015-01-01':'2015-12-31'].std())
print('')

# Plotting with distplot the Morgan Stanley returns 2015
print('2015 Morgan Stanley Returns')
sns.set_style('whitegrid')
sns.distplot(returns['MS Returns'].loc['2015-01-01':'2015-12-31'], color='green', bins=100)
plt.title('Morgan Stanley')
plt.show()

# Plotting with distplot the CitiGroup returns 2008
print('2008 CitiGroup Returns')
sns.distplot(returns['C Returns'].loc['2008-01-01':'2008-12-31'], color='r', bins=100)
plt.title('CitiGroup Returns')
plt.show()

# Create a line plot showing Close price for each bank for the entire index of time.

# Creating a for loop
print(bank_stocks)
for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12, 4), label=tick)
    plt.legend()
plt.title('For loop')
plt.show()

# Creating with cross section "xs"
bank_stocks.xs(key='Close', axis=1, level=1).plot(figsize=(12, 4))
plt.title('Cross Section')
plt.show()

# To Steps plot
close = bank_stocks.xs(key='Close', axis=1, level=1).reset_index()
print(close)
print('')
close.plot.line(x='Date', y='BAC C GS JPM MS WFC'.split(), figsize=(12, 6))
plt.title('2 Steps')
plt.show()

# Plotly
plotly.offline.plot(bank_stocks.xs(key='Close', axis=1, level=1).iplot(asFigure=True), filename='Stocks Plot.html')

# Create an average line against the close Price in Bank of America's 2008
plt.figure(figsize=(12, 4))
bank_stocks['BAC']['Close'].loc['2008-01-01':'2008-12-31'].rolling(window=30).mean().plot(label='30 Day moving Average')
bank_stocks['BAC']['Close'].loc['2008-01-01':'2008-12-31'].plot(label='BAC close price')
plt.legend()
plt.show()

# Creating the same plot with plotly

print(bank_stocks.loc['2008-01-01':'2008-12-31'].index)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        y=bank_stocks['BAC']['Close'].loc['2008-01-01':'2008-12-31'],
        x=bank_stocks.loc['2008-01-01':'2008-12-31'].index
    ))
fig.add_trace(
    go.Scatter(
        y=bank_stocks['BAC']['Close'].loc['2008-01-01':'2008-12-31'].rolling(window=30).mean(),
        x=bank_stocks.loc['2008-01-01':'2008-12-31'].index
    ))
plotly.offline.plot(fig, filename='BAC probe.html')

