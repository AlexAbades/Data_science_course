# Classification model: Predict the survival or desist
# Remember Kaggle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\13-Logistic-Regression"
                    r"\titanic_train.csv")

print(train.head())
print(train.info())

# First step is too see what and where are our missing data
# We can create a seaborn heatmap to see where it's
# We can pass the function isnull to pass a boolean filter, where the null values will become True

print(train.isnull().head())

# We don't want a color bar, so we indicate False, also we don't want the y labels, they would be row number
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
# To check the embarked null values, we don't need to use == True, because the values already are boolean values
print(train[train['Embarked'].isnull()])

# Cabin label has a lot of missing data, we should do something like:
# 1st: Drop it
# 2nd: Pass a boolean argument

sns.set(style='whitegrid')
# For classification problems it's always useful to see the ration, for example with count plots
sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')
plt.title('Survived')
plt.show()

# Survival ratios based on Sex
print('Survival ratios based on sex')
ratios= train.xs(key=['Survived', 'Sex', 'PassengerId'], axis=1).groupby(['Survived', 'Sex']).count()
print(ratios/train['PassengerId'].count()*100)

# Survival ratios based on Class
sns.countplot(x='Survived', hue='Pclass', data=train)
plt.show()
print('Survived ratios based on class')
print(train.xs(key=['Survived', 'Pclass', 'PassengerId'], axis=1).groupby(['Survived', 'Pclass']).count()/
      train['PassengerId'].count()*100)

# Lets see the distribution of the passengers age
# We can drop de the Nan Values with dropna()

sns.distplot(train['Age'].dropna(), kde=False, bins=30)
plt.title('Seaborn Visualization')
plt.show()

# It seems that it's like a by modal Distribution.
# We could have used the plot function inside pandas

train['Age'].hist(bins=30).plot()
plt.title('Pandas Visualization')
plt.show()

# Check the SibSp column, which indicates the number os children or spouse

sns.countplot(x='SibSp', data=train)
plt.title('SibSp')
plt.show()

# Check the Fare column

sns.distplot(train['Fare'])
plt.show()

import cufflinks as cf
from plotly.offline import plot

cf.go_offline()

plot(train['Fare'].iplot(kind='hist', bins=50, asFigure=True), filename='Titanic fare.html')