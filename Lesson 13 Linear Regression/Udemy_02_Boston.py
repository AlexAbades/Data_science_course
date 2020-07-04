import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Importing the data
from sklearn.datasets import load_boston

boston = load_boston()
print(boston)
print('')
# Checking the data, as we can se it's a dictionary, so we want to know first of all which are the keys.
print('Dictionary keys')
print(boston.keys())
print('')

# A brief description of our data:
# print(boston['DESCR'])

# Creating a DataFrame from boston
X = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
print(X)

y = pd.DataFrame(boston['target'], columns=['target'])
print(y)

# Creating the train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Instantiating an instance to our linear regression model, remember we are creating a linear regression object,
# so we can call functions into it

lm =LinearRegression()

# Fitting our model.
print('Model Fit')
print(lm.fit(X_train, y_train))
print('')

# Evaluating the model by checking its coefficients
print('Intercept values')
print(lm.intercept_)
print('')

# The intercept (often labeled the constant) is the expected mean value of Y when all X=0.

# Creating a Data Frame of the coefficient
print('Coefficient')
print(lm.coef_)
print(len(lm.coef_))
print(type(lm.coef_))
print('')
print('Column values')
print(X_train.columns)
print(len(X_train.columns))
print(type(X_train.columns))
print('')

print('Coefficient data Frame')
cfd = pd.DataFrame(data=lm.coef_, index=['Coef'], columns=X_train.columns).transpose()
print(cfd)
print('')
# Remember what does it mean the linear regression, like an equation. One increment in our variables it increase the
# coefficient value in our price, or target

# Now we have to get yhe predictions of our test
predictions = pd.DataFrame(data=lm.predict(X_test), columns=['predict'])
print('Prediction array')
print(predictions)
print(type(predictions))
print('')

# creating a scatter plot to view the regression
print('y_test')
print(y_test)
print('y_test type:')
print(type(y_test))


sns.set_style('whitegrid')
sns.scatterplot(x=y_test['target'], y=predictions['predict'])
plt.title('Boston regression model')
plt.show()


# view the value of our predictions with the real values




"""
# To make a real data analysis, we can make it in boston data set

from sklearn.datasets import load_boston

boston = load_boston()
print(boston)
print('')
# It's a dictionary, so we have to look out
print('dict keys')
print(boston.keys())
print('')
print('Description')
print(boston['DESCR'])
print('')
print('Columns of our data frame')
print(boston['feature_names'])
print('')
print('Data of the dictionary')
print(boston['data'])
print('')
print('Get the target prices in thousand')
print(boston['target'])
print('')


boston_df = boston.data
print(boston_df)
"""