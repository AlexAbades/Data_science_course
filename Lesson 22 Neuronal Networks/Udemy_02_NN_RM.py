import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\TensorFlow_FILES\DATA\kc_house_data.csv")

print(df.head())
print('\n')
print(df.info())
print('\n')
print(df.describe().transpose())
print('')

# To check if we have some miss data
print('Null Values')
print(df.isnull().sum())
print('\n')

sns.set_style('whitegrid')
plt.figure(figsize=(12, 8))
sns.distplot(df['price'])
plt.show()

# The points outside of th 95% of confidence, those prices that are higher than 1.5 million, we can asume that they
# are not usefull for our model

plt.figure(figsize=(12, 8))
sns.countplot(df['bedrooms'])
plt.show()

corr = df.corr()
print(corr['price'].sort_values())
print('\n')
plt.figure(figsize=(16,12))
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.show()

plt.figure(figsize=(16, 12))
sns.scatterplot(x='price', y='sqft_living', data=df)
plt.show()

f, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
sns.scatterplot(x='price', y='sqft_living', data=df, ax=axes[0, 0])
sns.scatterplot(x='price', y='grade', data=df, ax=axes[0, 1])
sns.scatterplot(x='price', y='sqft_above', data=df, ax=axes[1, 0])
sns.scatterplot(x='price', y='sqft_living15', data=df, ax=axes[1, 1])
plt.show()

plt.figure(figsize=(16, 12))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title('Bedrooms')
plt.show()

plt.figure(figsize=(16, 12))
sns.boxplot(x='floors', y='price', data=df)
plt.title('Floors')
plt.show()

plt.figure(figsize=(16, 12))
sns.boxplot(x='bathrooms', y='price', data=df)
plt.title('Bathrooms')
plt.show()

plt.figure(figsize=(16, 12))
sns.boxplot(x='view', y='price', data=df)
plt.title('View')
plt.show()

g = sns.PairGrid(data=df, height=8, aspect=2, x_vars=['price'], y_vars=['long', 'lat'])
g.map(plt.scatter, edgecolor='white')
plt.show()

# king county map

plt.figure(figsize=(16, 12))
sns.scatterplot(x='long', y='lat', data=df, hue='price')
plt.show()

print('Top 20 more expensive houses')
print(df.sort_values('price', ascending=False)['price'].head(20))
print('\n')

print('1% of the length of our data frame')
print(len(df)*0.01)
print('\n')

#  So if we order our data frame, and we drop the first 1%, we will drop the 1% more expensive houses

non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]
print(non_top_1_perc.sort_values('price', ascending=False)['price'].head(20))
print('\n')

plt.figure(figsize=(16, 12))
sns.scatterplot(x='long', y='lat', data=non_top_1_perc, hue='price',
                edgecolor= None, alpha=0.2, palette='RdYlGn')
plt.show()


plt.figure(figsize=(16, 12))
sns.boxplot(x='waterfront', y='price', data=df)
plt.ylabel('Price')
plt.ylabel('Waterfront')
plt.legend(['NO', 'YES'])
plt.title('Waterfront vs Price')
plt.show()

print('Original Data base')
print(df.head())
print('\n')

df = df.drop('id', axis=1)

# checking the date column

print('Date column')
print(df['date'])
print('\n')
# Notice that the date column it's actually an object, not a datetime object

# We can convert the date column into a DateTime object

df['date'] = pd.to_datetime(df['date'])

print('DateTime Object')
print(df['date'])
print('\n')

df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

print('New date Frame')
print(df)
print('\n')

plt.figure(figsize=(16, 12))
ax = sns.boxplot(x='month', y='price', data=df)
plt.show()

by_month = df.groupby('month').mean()['price']
print(by_month)
print('\n')
plt.figure(figsize=(16, 12))
by_month.plot()
plt.show()

plt.figure(figsize=(16, 12))
sns.lineplot(data=by_month)
plt.show()

by_year = df.groupby('year').mean()['price']
print(by_year)
print('\n')

plt.figure(figsize=(16, 12))
by_year.plot()
plt.show()

# As we've get the date information from the date column, we can drop it

df = df.drop('date', axis=1)

# The zip code column is a number, anf if we don't do anything, our model it's going to assume that in some sort of
# continues features.

# If there isn't a continue relation, we will treat them as categorical feature

print(df['zipcode'].value_counts())
print('\n')
# As we have around 70 zip codes, in this particular case we're going to drop de column. But In a real case, we will
# want to plot this, by mapping them manually

df = df.drop('zipcode', axis=1)

# The other column or feature that we've to be concern it, is the year of renovation.

print(df['yr_renovated'].value_counts())
print('\n')
# And as there are a lot of non renovation values, we'll probably want to treat them as a categorical values.
# Renovated or Not renovated

df['Renovated'] = df['yr_renovated'].apply(lambda year: 1 if year != 0 else 0)

# In this case, the value increases as the renovation year is more recent, so we don't have to drop it. Because it
# has sense to keep it as a continuous variable

print(df)
print('\n')

from sklearn.model_selection import train_test_split

X = df.drop('price', axis=1).values

y = df['price'].values

# Tensorflow won't accept anything that isn't a numeric array, it can't work with pandas series or data frames.
# Check that our split matrix are numpy arrays, with .vlaues
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

# We don't fit on our test set

X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

# We try to base the number of neurons or units in our layers from the size of the actual feature data
print(X_train.shape)

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Regression model, we are selecting a continuous label such price, we are going to use the mean squared error loss
# function

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          batch_size=128,
          epochs=800)

# Validation Data: After each epoch of training on the training data will quickly run the test data and check our
# loss on the test data. So in that way we can keep a tracking of how well performing, not just on our training data,
# but also in our test data. Keep in mind that these test data will not actually affect the weight or biases of ous
# network So keras is not going to update the model based off the test data or validation data. Instead in will only
# use the training data to upload the weight and biases.

# Batches_size: Because it's a large Data set, we're gonna fit our data in batches (a set of N samples). Topically to
# set the batches sizes in powers of two, so, 64, 128, 256... The smaller size the longer training it's going to take
# but less likely you're going to over fit to your data because you're not passing your entire data set at once,
# instead you're focusing on this smaller batches

losses = pd.DataFrame(model.history.history)
# Val_loss: Loss on the test set
losses.plot()
plt.show()
# If the validation line on the plot begins to increase, after the both of them had decrease. It's a bad indicator.
# Early stopping

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error: ', mse, '\n')
rmse = np.sqrt(mse)
print('Root Mean Squared Error: ', rmse, '\n')
mae = mean_absolute_error(y_test, predictions)
print('Mean Absolute Error: ', mae, '\n')

print(df.describe)

# Our mean price for houses it's 540296.6, so our MAE it's about a 20% of the value...
# We can call explained variance to take conclusions of what our model it's explaining

print(explained_variance_score(y_test, predictions))

plt.figure(figsize=(16, 12))
plt.scatter(y_test, predictions, edgecolors='white')
plt.plot(y_test, y_test, color='r')
plt.show()

# Brand new house

single_house = df.drop('price', axis=1).iloc[0]

# We have to scale the features

# We first grab our values and convert it into a np array

print(single_house.values)
print('\n')
# If we pay attention to the shape, we only will see one square bracket, we have to reshape it

print('Reshaped')
single_house = single_house.values.reshape(-1, 20)
# (-1,19) just basically means keep those old dimensions along the axis

single_house = scaler.transform(single_house)
print(model.predict(single_house))

print(df['price'].iloc(0))

##################################################################################################
"""
REPEAT THE PROCESS WITHOUT THAT 1% TOP EXPENSIVE HOUSES
"""
##################################################################################################

