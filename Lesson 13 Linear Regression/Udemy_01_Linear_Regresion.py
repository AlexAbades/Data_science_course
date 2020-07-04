import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# We import the file
df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp"
                 r"-master\11-Linear-Regression\USA_Housing.csv")

# We check the head of the dataframe
print(df.head())
print('')
# check the info
print(df.info())
print('')
# Check describe method in order to get a quick account of some statistical information for numerical values
print(df.describe())
print('')
# Getting the column names
print(df.columns)

# Creating a quick visualization
sns.pairplot(df)
plt.show()

# Distribution plot of the price
sns.set_style('whitegrid')
sns.distplot(df['Price'])
plt.show()

# Heat map of the correlation
sns.heatmap(data=df.corr(), cmap='coolwarm', annot=True)
plt.show()

# Using scikit and train a linear regression

# 1st: The first we need to do is split our data into a X array that contains the features to train on and a Y array
# with a target variable; in this case the price column which is what we are trying to predict
# So, X it's going to have all the columns except the price, because it's what we are trying to predict and the address
# column because it's a string

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]
print(type(X))
y = df['Price']

# Once we have our x and y data, the next step it's to do our train test split on our data. We have to import the
# function

from sklearn.model_selection import train_test_split

# The random_state argument it's only to make that our random selection of the data, it's the same for us that for
# the professor, because the train_test_split function makes a random selection pf all the data base for the test and
# train variables. It that way we get the same results that the teacher.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
print('X_train')
print(X_train)
print('')
print('X_test')
print(X_test)
print('')
print('y_train')
print(y_train)
print('')
print('y_test')
print(y_test)
print(type(y_test))
print('')

# The next step is to create and train the model, so we have to import the Linear Regression Model

from sklearn.linear_model import LinearRegression

# instantiate = ejemplificar;  instance = instancia, caso, ejemplo
# Once if imported the linear regression model we go ahead and instantiate an instance of the Linear Regression Model

lm = LinearRegression()  # Instantiate it, so we are creating a Linear Regression Object

# The first method that we're going to call it's .fit to train or fit my model, in this case I only want a fit my on
# my training data

# This gives us an out put telling us that the linear regression has been Trained
lm.fit(X_train, y_train)
# Notice that we don't set the lm.fit to any other variable object. Its already taking effect onto the object itself.
# So we don't have to set lm = lm.fit

# Now, we can evaluate our model by checking out its coefficients and then seeing how we can interpret them

# Printing the intercept:
print(lm.intercept_)

# The next we can do, is check the coefficients and the coefficient are going to relate to each feature in our data set
print(lm.coef_)
# The coefficients are related to the columns in x.train
print(X_train.columns)

# Creating a Data frame to see better our coefficients

cdf= pd.DataFrame(data=lm.coef_, index=X_train.columns, columns=['Coeff'])
print(cdf)
# As a linear regression it creates an theoretical equation like y = X*n +e being the "e" the error. So the matrix
# says or means that if all the other features are fixed, a one unit increase in Avg. Area Income it would increase
# would increase 21.528276$ in the price. it would be y = 25.528276x being y the price and x the area income

# Getting predictions from our model
# Predictions
predictions = lm.predict(X_test)
print('Predictions')
print(predictions)
print(type(predictions))
print('')
# This are the predictions results, we are going compare them with the real results that we already have (y_test)
print('Real Test')
print(y_test)

# Creating a scatter plot for visualize the prediction
# Verifying the type
print(type(y_test))
print((type(predictions)))
sns.set_style('whitegrid')
sns.scatterplot(x=y_test, y=predictions)
plt.show()
# A perfect straight line it would be a perfectly correct predictions, so a little bit off the sort of straight line
# is actually a very good job.

# Create a histogram of our residuals (the error, the difference between the actual values and the predicted values)
sns.distplot((y_test-predictions))
plt.show()
# Notice that it's a normal distribution, it's a good sign that what we've chose it's a correct choice for our data.

# Regression evaluation matrix
# MEA = Mean Absolute Error,
# MSE = Mean Squared Error
# RMSE = Root Mean Squared Error
# For more info check the ppt, but the point is to minimize this values

from sklearn import metrics

MEA = metrics.mean_absolute_error(y_test, predictions)
print(MEA)
MSE = metrics.mean_squared_error(y_test, predictions)
print(MSE)
RSME = np.sqrt(metrics.mean_squared_error(y_test, predictions))
