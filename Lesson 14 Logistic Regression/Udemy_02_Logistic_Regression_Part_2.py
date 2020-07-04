import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\13-Logistic-Regression"
                    r"\titanic_train.csv")

print(train.info())
# Cleaning our data

sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.show()

# We can use different methods
# We can fill in with the mean age of al the passengers, which it's known as IMPUTATION
# but we can create a box plot by class and by sex

sns.set(style='whitegrid')
plt.figure(figsize=(10, 7))
sns.boxplot(y='Age', x='Pclass', data=train, hue='Sex')
plt.show()

average_age = train.xs(key=['Pclass', 'Sex', 'Age'], axis=1).groupby(['Pclass', 'Sex']).mean()
print(average_age)
print(average_age.loc[(1, 'male')])


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]

    if pd.isnull(Age):
        if Pclass == 1 and Sex == 'male':
            return average_age.loc[(1, 'male')]
        elif Pclass == 1 and Sex == 'female':
            return average_age.loc[(1, 'female')]
        elif Pclass == 2 and Sex == 'male':
            return average_age.loc[(2, 'male')]
        elif Pclass == 2 and Sex == 'female':
            return average_age.loc[(2, 'female')]
        elif Pclass == 3 and Sex == 'male':
            return average_age.loc[(3, 'male')]
        else:
            return average_age.loc[(3, 'female')]
    else:
        return Age


train['Age'] = train[['Age', 'Pclass', 'Sex']].apply(impute_age, axis=1)
# As we can see we have a lot of missing data in cabin, so the best thing that we can do, it's drop it.

train.drop(labels=['Cabin'], axis=1, inplace=True)

sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.show()

# We still have a little rows in Embarked column without information, so we can drop these null values

train.dropna(inplace=True)

# So the next step it's to make dummy variables for the categorical variables
# we can make it in our own, or we can make it with the "get_dummies" function

print(pd.get_dummies(train['Sex']))

# It creates a data frame with a column for every single category, and a zero or one value which is essentially a
# boolean filter, or very similar. As we have two columns, the first one it'a a perfect predictor of the second one.
# And that means that if our machine learning algorithm gets fed both columns. That's an issue known as
# multi-collinearity and it basically mess up the algorithm because the a bunch of columns will be perfect predictors
# of a another column in order to avoid this, we can drop one of the columns.

print(pd.get_dummies(train['Sex'], drop_first=True))

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

# We can use concatenate to create a knew data frame

train = pd.concat([train, sex, embark], axis=1)
print(train.head())

# We can try to use the other features, like the title (Mr, Mss, or others), but at that point we are going to drop them

train.drop(labels=['Embarked', 'Sex', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
print(train.head())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print('Predictors')
X =train.drop('Survived', axis=1)
print(X.head())

print('Target')
y = train['Survived']
print(y.head())


print(type(train['Fare']))


# then we separate our predictors and target data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Create the model
lgm = LogisticRegression(max_iter=5000)

# Train the model
lgm.fit(X_train, y_train)


# We create the prediction

predictions = lgm.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

# We can use a clarification report
print(classification_report(y_test, predictions))

# Or we can use the confusion matrix

print(confusion_matrix(y_test, predictions))


