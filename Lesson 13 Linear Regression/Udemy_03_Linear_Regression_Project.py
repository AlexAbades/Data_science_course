import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Read the customers csv file
customers = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\11-Linear-Regression"
                        r"\Ecommerce Customers")

# 1 check the head of the data Frame
print('Ex 1; head')
print(customers.head())
print('Info')
print(customers.info())
print('Describe')
print(customers.describe())

# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the
# correlation make sense?

sns.set_style('whitegrid')
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', edgecolor='black', linewidth= 0.5,
              data=customers, color='grey')
plt.title('TimeWeb/Spend')
plt.show()


#  Do the same but with the Time on App column instead.

sns.jointplot(x='Time on App', y='Yearly Amount Spent', edgecolor='black', linewidth=0.5,
              data=customers, color='grey')
plt.title('TimeApp/Spend')
plt.show()

# Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.

sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex', color='grey')
plt.title('Length/TimeApp')
plt.show()

# Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.(
# Don't worry about the the colors)

sns.pairplot(customers)
plt.show()

# Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers, palette='coolwarm')
plt.title('Linear Regression length vs Spent')
plt.show()

# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets. ** Set a
# variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent"
# column.

X = customers._get_numeric_data().drop('Yearly Amount Spent', axis=1)
print('Numerical data')
print(X)

print('Target: total amount spent')
y = customers['Yearly Amount Spent']
print(y)

# Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set
# test_size=0.3 and random_state=101
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)

# Training our model.

#  Import LinearRegression from sklearn.linear_model

from sklearn.linear_model import LinearRegression

# Create an instance of a LinearRegression() model named lm
lm = LinearRegression()

# Train/fit lm on the training data
print(lm.fit(X_train, y_train))

# Intercept the value of y at x equal 0
print(lm.intercept_)

# Coefficients
print(lm.coef_)
cdf = pd.DataFrame(lm.coef_, index=X_train.columns, columns=['Coef'] )
print(cdf)
print('')

# Predicting data
prediction = lm.predict(X_test)

# Create a Scatterplot of the real test values versus the predicted values.

sns.scatterplot(x=y_test, y=prediction, palette='Blue')
plt.xlabel('Y Test (True Values')
plt.ylabel('Predicted Values')
plt.show()

# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to
# Wikipedia for the formulas

from sklearn import metrics

MAE = metrics.mean_absolute_error(y_test, prediction)
MSE = metrics.mean_squared_error(y_test, prediction)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, prediction))
Exp_var= metrics.explained_variance_score(y_test, prediction)
print('Explained Variance ', Exp_var)
print('MAE: ', MAE)
print('MSE: ',MSE)
print('RMSE: ', RMSE)

# Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot,
# or just plt.hist().

sns.distplot((y_test-prediction), color='green', bins=50)
plt.show()

