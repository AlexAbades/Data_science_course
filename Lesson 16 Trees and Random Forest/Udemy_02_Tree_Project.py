import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

loans = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\15-Decision-Trees-and-Random"
                    r"-Forests\loan_data.csv")

print(loans)
print('\n')
print(loans.info())
print('\n')
print(loans.describe())

# Distplot or hist plot
sns.set_style('whitegrid')
sns.distplot(loans[loans['credit.policy'] == 1]['fico'], kde=False, label='Credit.Policy=1', color='purple',
             hist_kws={'alpha': 0.5}, bins=30)
sns.distplot(loans[loans['credit.policy'] != 1]['fico'], kde=False, label='Credit.Policy=0', color='r', bins=30)
plt.legend()
plt.show()

loans[loans['credit.policy'] == 1]['fico'].hist(bins=30, color='red', alpha=0.5, label='credit.policy=1')
loans[loans['credit.policy'] != 1]['fico'].hist(bins=30, color='purple', alpha=0.5, label='credit.policy=0')
plt.legend()
plt.show()

sns.distplot(loans[loans['not.fully.paid'] == 0]['fico'], kde=False, bins=30, color='red', hist_kws={'alpha':0.5},
             label='Not.fully.paid=0')
sns.distplot(loans[loans['not.fully.paid'] != 0]['fico'], kde=False, bins=30, label='not.fully.paid=1',
             color='purple', hist_kws={'alpha':0.5})
plt.xlabel('FICO')
plt.legend()
plt.show()

sns.countplot(x='purpose', data=loans, hue='not.fully.paid', palette=['red', 'blue'])
plt.show()

sns.jointplot(x='fico', y='int.rate', data=loans, color='mediumorchid', edgecolor='black', linewidth=0.2)
plt.show()

sns.lmplot(x='fico', y='int.rate', col='not.fully.paid', hue='credit.policy', data=loans, palette='Set1')
plt.show()

# Creating our model, we have to create a dummy variable for the purpose column
# The column has to pass a list variable. We could create a variable with a list and pass it there

final_data = pd.get_dummies(loans, columns=['purpose'], drop_first=True)

print(final_data.info())

# train test

X = final_data.drop('not.fully.paid', axis=1)

y = final_data['not.fully.paid']

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

rfc = RandomForestClassifier(n_estimators=300)

rfc.fit(X_train, y_train)

predict = rfc.predict(X_test)

print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))


