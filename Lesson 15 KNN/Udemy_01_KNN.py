import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\14-K-Nearest-Neighbors\Classified "
                 r"Data", index_col=0)

print(df.head())
print('')
print(df.info())

# As we are going to use the KNN which uses the distance of nearest neighbours, the scale of the variables matters a
# lot, so the first step is to standardise all the data, Sklearn has a very useful tool

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# We create an instance

scaler.fit(df.drop('TARGET CLASS', axis=1))
# We created an scalar object

# Then we use the scalar object to create a transformation
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
# Performs standardization by centering and scaling
print(scaled_features)
# It's going to be an array of values, scaled version

df_feat = pd.DataFrame(scaled_features, columns=df.columns[0:-1])
print(df_feat.head())

# Know the data it's ready to use machine learning

X = df_feat  # We could pass the array of values: scaled_features

y= df['TARGET CLASS']

X_test, X_train, y_test, y_train = train_test_split(X, y, random_state=101, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(pred)

# Evaluation

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))
print('')

print(classification_report(y_test, pred))

# The model it's pretty good, but we are going to see if we can choose a better k value with elbow method
# It's going to iterate through k values to find which has the lowest error rate

error_rate = []

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i= knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


print(error_rate)

plt.figure(figsize=(10, 6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.show()

# We create a for loop for 1 to 40, then we call oir knn model, where the number of neighbours are going to be the
# values from 1 to 40 (the for loop). Then we create a predictor for each value of k. Where we are going to calculate
# the average where our predicted values are not equal to the true values


knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
predict = knn.predict(X_test)
print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))