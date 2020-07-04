import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ex 1 Import scv file
print('Ex 1')
df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\10-Data-Capstone-Projects\911.csv")
print(df)
print('')

# Ex 1: Check the Head

print('Ex 2')
print(df.head())
print('')

# Ex 3: Check the info type...
print('Ex 3')
print(df.info())
print('')

# Ex 4: Top 5 Zip code
print('Ex 4')
print(df['zip'].value_counts().head())
print('')

# Ex 5: Top 5 townships code
print('Ex 5')
print(df['twp'].value_counts().head())
print('')

# Ex 6: How many unique titles are
print('Ex 6')
print(df['title'].nunique())
print('')

# Ex 7: How many unique titles are
print('Ex 6')
print(df['title'].nunique())
print('')

# Ex 8: In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire,
# and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this
# string value.
print('Ex 6')
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
print(df)
print('')

# Ex 9: What is the most common Reason for a 911 call based off of this new column?
print('Ex 9')
print(df['Reason'].value_counts())
print('')

# Ex 10:  Now use seaborn to create a countplot of 911 calls by Reason
print('Ex 10')
sns.set_style('whitegrid')
sns.countplot(x='Reason', data=df, palette='rainbow')
plt.title('Ex 10')
plt.show()
print('')

# Ex 11: Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column
print('Ex 11')
print(type(df['timeStamp'][0]))
print('')
print(df.timeStamp.dtype)
print('')
print(df['timeStamp'].loc[0])

# Ex 12:  You should have seen that these timestamps are still strings. Use pd.to_datetime to convert the column from
# strings to DateTime objects
print('Ex 12')
df['timeStamp'] = pd.to_datetime(df['timeStamp'])  # https://pandas.pydata.org/pandas-docs/stable/reference/api
# /pandas.to_datetime.html We can specify more arguments.
time = df['timeStamp'].iloc[0]
print(time)
print(time.day)  # hour, minute, second, day, month, year...
print('')

# Ex 13: You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp
# column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You
# will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.
print('Ex 13')

# Creating a new column for hour
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
print(df['Hour'])

# Creating a column for months
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
print(df['Month'])

# Creating a new column for days, mapping
dmap = {0: 'Mon', 1: 'Tue', 2: 'Wen', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
print('dictionary', dmap)
print('Type \n', type(time.dayofweek))
# In one step
df['Day'] = df['timeStamp'].apply(lambda x: x.dayofweek).map(dmap)
print(df['Day'])

# Ex 14: Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.
print('Ex 14')
sns.countplot(x='Day', data=df, hue='Reason')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Ex 14')
plt.show()
print('')

# Ex 15: Now do the same for Month
sns.countplot(x='Month', data=df, hue='Reason', palette='viridis')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Ex 15')
plt.show()
print('')

# Ex 16: Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the
# count() method for aggregation. Use the head() method on this returned DataFrame

# Line plot
print('Ex 16')
byMonth = df.groupby('Month').count()
print(byMonth.head())
byMonth['lat'].plot()
plt.title('Ex 16')
plt.show()
print('')

# Ex 17: Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in
# mind you may need to reset the index to a column
print('Ex 17')
sns.lmplot(x='Month', y='twp', data=byMonth.reset_index())  # It seems that when we specify it inside the lmplot it
# works
plt.title('Ex 17.1')
plt.show()

# We could make it also like..
# byMonth['Month'] = byMonth.index
byMonth = byMonth.reset_index()  # We have to assign back to the variable
print(byMonth.head())
# sns.lmplot(x='Month', y='lat', data=byMonth)
# plt.xlim(0, 14)
# plt.show()

print('')

# Ex 18: Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply
# along with the .date() method.
df['Date'] = df['timeStamp'].apply(lambda x: x.date())
date = df['Date'].iloc[0]
print(df, '\n', date)
print('')

# Ex 19: Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls
print('Ex 19')
byDate = df.groupby('Date').count()
print(byDate)
byDate['lat'].plot()
plt.title('Ex 19')
plt.show()
print('')

# An other way to do it
byDate1 = df.groupby('Date').count()['lat'].plot()
plt.title('Ex 19.2')
plt.show()

# Ex 20: Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call
print('Ex 20: EMS')
df[df['Reason'] == 'EMS'].groupby('Date').count()['lat'].plot()
plt.title('Ex 20: EMS')
plt.show()
print('')

print('Ex 20: Fire')
df[df['Reason'] == 'Fire'].groupby('Date').count()['lat'].plot()
plt.title('Ex 20: Fire')
plt.show()
print('')

print('Ex 20: Traffic')
df[df['Reason'] == 'Traffic'].groupby('Date').count()['lat'].plot()
plt.title('Ex 20: Traffic')
plt.show()

# Ex 21: Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the
# dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to
# do this, but I would recommend trying to combine groupby with an unstack method. Reference the solutions if you get
# stuck on this!
print('Ex 21')
print(df.info())
matrix = df.groupby(['Day', 'Hour']).count()['lat'].unstack(level=-1)
print(matrix)

# Sorting the rows
sorter = 'Mon Tue Wen Thu Fri Sat Sun'.split()
sorterIndex = dict(zip(sorter,range(len(sorter))))
print(sorterIndex)
matrix['Day_index'] = matrix.index
print(matrix)
matrix['Day_index'] = matrix['Day_index'].map(sorterIndex)
print(matrix)
matrix.sort_values('Day_index', inplace=True)
matrix.drop('Day_index', axis=1, inplace=True)
print(matrix)
sns.heatmap(matrix, cmap='Greens')
plt.show()
# We could use a pivot or a pivot table of pandas to make a matrix. Can we order the matrix?


# Cluster Map
sns.clustermap(matrix)
plt.show()

#Changeing the matrix to day/month

mat = df.groupby(['Day', 'Month']).count()['lat'].unstack(level=-1)
mat['Index day'] = mat.index.map(sorterIndex)
mat.sort_values('Index day', inplace=True)
mat.drop('Index day', axis=1, inplace=True)
print(mat)

# Heat map
sns.heatmap(mat, cmap='viridis')
plt.title('Heat Map')
plt.show()

#Clustar map
plt.figure(figsize=(12,8))
sns.clustermap(mat, cmap='viridis')
plt.title('Cluster Map')
plt.show()