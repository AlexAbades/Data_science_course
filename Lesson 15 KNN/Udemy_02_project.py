import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\14-K-Nearest-Neighbors"
                  r"\KNN_Project_Data")
print(df.head())
print('\n')
print(df.info())

# sns.pairplot(df, hue='TARGET CLASS')
# plt.show()

# We crate an object
scaler = StandardScaler()

# We fit the object
scaler.fit(df.drop('TARGET CLASS', axis=1))

# We make the transformation
scaler_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

print(scaler_features)

# We create the data frame

df_feat = pd.DataFrame(scaler_features, columns=df.columns[:-1])

print(df_feat)

# We create our train test split
X = df_feat

y = df['TARGET CLASS']


X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3, random_state=101)

# We create our model

knn = KNeighborsClassifier(n_neighbors=1)

# We fit our model
knn.fit(X_train, y_train)

# We make the predictions

pred = knn.predict(X_test)

# We call the confusion matrix and the classifier report

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

# elbow method

error_rate = []
for i in range(1, 50):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Elbow analysis for K')
plt.xlabel('Nº Of K')
plt.ylabel('Error rate')
plt.show()

# k= 32 & k= 12
# K = 12
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
pred1 = knn.predict(X_test)
print(confusion_matrix(y_test, pred1))
print('\n')
print(classification_report(y_test, pred1))

# K = 32

knn = KNeighborsClassifier(n_neighbors=32)
knn.fit(X_train, y_train)
pred2 = knn.predict(X_test)
print(confusion_matrix(y_test, pred2))
print('\n')
print(classification_report(y_test, pred2))


