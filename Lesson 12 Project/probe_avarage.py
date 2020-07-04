import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf

bank_stocks = pd.read_pickle(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\10-Data-Capstone"
                             r"-Projects\all_banks")

BAC = bank_stocks['BAC']['Close'].loc['2008-01-01':'2008-12-31']

mean = []
count=0
for element in BAC:
    if count >= 30:
        mean.append(BAC[count-30:count+1].mean())
    count+=1
print(mean)
# Changing the data.Series to data.Frame
BAC = pd.DataFrame(BAC)
print('BAC Type:')
print(type(BAC))
# Creating a new column
BAC['Join'] = np.arange(0, len(BAC.index))
print(BAC)
BAC = BAC.reset_index()
print('')

# Creating a data frame from mean list
BAC_mean = pd.DataFrame(mean, columns=['BAC mean'])
print(BAC_mean)
print('')

# Creating a new column based on the mean results
BAC_mean['Join'] = np.arange(30, len(BAC.index))
print(BAC_mean)

# Merging the two data frames
bac = pd.merge(BAC, BAC_mean, on='Join', how='left', left_index=True)
print(bac)

# Setting the index as Date
bac.set_index('Date', inplace=True)
print(bac)

# Creating the plot
sns.set_style('whitegrid')
bac['Close'].plot(figsize=(12, 4))
bac['BAC mean'].plot(figsize=(12, 4))
plt.show()

#Creating Plot with plotly
bac.drop('Join', axis=1,inplace=True)
print(bac)

plot(bac.iplot(asFigure=True),filename='BAC.html')

# https://stackoverflow.com/questions/14102498/merge-dataframes-different-lengths