from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# check the link: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html

"""
We also can use the following method
import pandas_datareader.data as web
BAC=web.DataReader('BAC', 'stooq')
print(BAC)
"""

# BANK OF AMERICA

start = datetime.datetime(2006, 1, 1)  # As all the info we are going to catch is from the same period of time,
# we haven't to specify it each time
end = datetime.datetime(2016, 1, 1)

BAC = data.DataReader("BAC", 'yahoo', start, end)

# CITIGROUP

C = data.DataReader('C', 'yahoo', start, end)

# GOLDAN SACHS GROUP INC

GS = data.DataReader('GS', 'yahoo', start, end)

# JP MORGAN CHASE

JPM = data.DataReader('JPM', 'yahoo', start, end)

# MORGAN STANLEY

MS = data.DataReader('MS', 'yahoo', start, end)

# WELLS FARGO

WFC = data.DataReader('WFC', 'yahoo', start, end)

# Create a list of the ticker symbols (as strings) in alphabetical order. Call this list: tickers

tick = [BAC, C, GS, JPM, MS, WFC]
tickers = 'BAC C GS JPM MS WFC'.split()

# Use pd.concat to concatenate the bank dataframes together to a single data frame called bank_stocks. Set the keys
# argument equal to the tickers list. Also pay attention to what axis you concatenate on.

bank_stocks = pd.concat(objs=tick, axis=1, keys=tickers)
# 1st. The objects must be objects, as we can see the difference between the tick and tickers variable, one it's a
# list of actual data, and the other is a list of strings.
# 2nd. The keys argument, allows to make a multi index level, as the axis it's 1, it will create it in the columns.

# Set the column name levels
bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']
print(bank_stocks)

print(bank_stocks.xs(key='Close', axis=1, level=1).max())
print('')  # level = Stock info = 1

# An other Way to do it it's with a for lop.
for item in tickers:
    print(bank_stocks[item]['Close'].max())

# https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.xs.html
# https://jakevdp.github.io/PythonDataScienceHandbook/03.05-hierarchical-indexing.html

# We can use pandas pct_change() method on the Close column to create a column representing this return value.
# Create a for loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in
# the returns DataFrame.

# Easy way, the equation it really return: Percentage change between the current and a prior element.
returns = bank_stocks.xs(key='Close', axis=1, level=1).pct_change()
print(returns)

# Creating a for loop

# 1st we create the column names, at the same time it will help to iterate through the loop
ret_tickers = 'BAC Return,C Return,GS Return,JPM Return,MS Return,WFC Return'.split(',')
print(ret_tickers)
print('')
print('NEXT')
print('')

# 2nd We create the data Frame returns1, we couldn't set the columns with only the number of columns,
returns1 = pd.DataFrame(columns=ret_tickers)

# Checking the info
print(returns1.info())
print(bank_stocks['BAC']['Low'])
print('')

# To loop through two lists at the same time, we have to zip the lists
tickers = 'BAC C GS JPM MS WFC'.split()
print(list(zip(ret_tickers, tickers)))

# Creating a for loop through a zip
for new, old in zip(ret_tickers, tickers):
    returns1[new] = bank_stocks[old]['Close'].pct_change()
# An other method:
new_matrix = pd.DataFrame()
for i in tickers:
    new_matrix[i + ' Return'] = bank_stocks[i]['Close'].pct_change()
# Printing the return Matrix
print(returns1)

sns.pairplot(data=new_matrix[1:])
plt.show()

# Probes with csv

all_banks = pd.read_pickle(
    r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\10-Data-Capstone-Projects\all_banks")
print(all_banks.head())
a =dict(zip(tickers, ret_tickers))
all_banks_return = all_banks.xs(key='Close', axis=1, level=1).pct_change()
print('All_Banks')
print(all_banks_return.head())
print(a)
print('')
all_banks_return.rename(columns={"BAC": "BAC Returns"})
print(all_banks_return)
sns.pairplot(data=all_banks_return)
plt.show()