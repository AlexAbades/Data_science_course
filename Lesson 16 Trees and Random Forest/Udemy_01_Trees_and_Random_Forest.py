import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn. metrics import classification_report, confusion_matrix

df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\15-Decision-Trees-and-Random"
                 r"-Forests\kyphosis.csv")

print(df.head())
print('')
print(df.info())

sns.set_style('whitegrid')
sns.pairplot(df, hue='Kyphosis')
plt.show()

X = df.drop('Kyphosis', axis=1)

y = df['Kyphosis']

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

# Now we are going to compare to the random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))
print('\n')
print(classification_report(y_test, rfc_pred))

# The data set it's unbalanced, so that the motive it's better the tre that the random forest
