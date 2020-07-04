import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ad_data = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\13-Logistic-Regression"
                      r"\advertising.csv")
print(ad_data.head())
print('')

print(ad_data.info())
print('')

print(ad_data.describe())

sns.set(style='whitegrid')

# Age histogram
sns.distplot(ad_data['Age'], bins=30)
plt.title('Age')
plt.show()

# Joint plot average income vs age

sns.jointplot(y='Area Income', x='Age', data=ad_data, edgecolor='black', linewidth=0.35)
plt.show()

sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde', color='r')
plt.show()

sns.jointplot(x='Daily Internet Usage', y='Daily Time Spent on Site', data=ad_data, color='g',
              marginal_kws=dict(bins=20), edgecolor='black', linewidth=0.3)
plt.show()

# sns.pairplot(ad_data, hue='Clicked on Ad', palette='RdBu_r', diag_kind='hist')
# plt.show()

X = ad_data.drop(['Clicked on Ad', 'Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)
print(X)
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)

lgm= LogisticRegression(max_iter=5000)

lgm.fit(X_train, y_train)

predictions = lgm.predict(X_test)

print('Classification report')
print(classification_report(y_test, predictions))
print('')

print('Confusion matrix')
print(confusion_matrix(y_test, predictions))

