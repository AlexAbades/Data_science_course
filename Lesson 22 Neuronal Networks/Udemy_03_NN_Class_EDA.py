# Classification task with tensorflow
# How to identify and deal with overfitting through Early Stopping Callbacks and Dropout Layers

# Early Stopping:
# Keras can automatically top training based on a lost condition on the validation data that we pass
# in during the model that model.fit call

# Dropout Layers:
# Dropout can be added to layers to "turn off" neurons during training to prevent overfitting. Basically what we do is
# each drop layer will drop or turn off a user defined percentage of neurons units in the previous layer every batch
# So that means certain neurons don't have their weights or biases affected during a batch, instead their are just
# turned off

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\TensorFlow_FILES\DATA\cancer_classification.csv")

print('Data Frame')
print(df)
print('\n')

print('INFO')
print(df.info())
print('\n')

print('Describe')
print(df.describe().transpose())
print('\n')

print('Null values')
print(df.isnull().sum())
print('\n')

plt.figure(figsize=(16, 16))
sns.set_style('whitegrid')
sns.countplot(x='benign_0__mal_1', data=df)
plt.show()

corr = df.corr()
print('Correlation')
print(corr['benign_0__mal_1'].sort_values())
print('\n')

plt.figure(figsize=(16, 12))
sns.heatmap(data=corr, annot=True, cmap='coolwarm')
plt.show()

print(df.columns)

plt.figure(figsize=(16, 12))
corr['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')  # Instead of grab[:-1] we couls use
# .drop('benign_0__mal_1')
plt.show()


f, ax = plt.subplots(3, 3, figsize=(16, 12))
sns.scatterplot(x='mean radius', y='mean area', data=df, hue='benign_0__mal_1', ax=ax[0, 0])
sns.scatterplot(x='mean radius', y='perimeter error', data=df, hue='benign_0__mal_1', ax=ax[0, 1])
sns.scatterplot(x='mean area', y='worst radius', data=df, hue='benign_0__mal_1', ax=ax[0, 2])
sns.scatterplot(x='mean compactness', y='worst compactness', data=df, hue='benign_0__mal_1', ax=ax[1, 0])
sns.scatterplot(x='mean perimeter', y='worst concave points', hue='benign_0__mal_1', data=df, ax=ax[1, 1])
sns.scatterplot(x='mean perimeter', y='mean concave points', data=df, hue='benign_0__mal_1', ax=ax[1, 2])
sns.scatterplot(x='mean concave points', y='worst texture', data=df, hue='benign_0__mal_1', ax=ax[2, 0])
sns.scatterplot(x='mean symmetry', y='mean compactness', data=df, hue='benign_0__mal_1', ax=ax[2, 1])
sns.scatterplot(x='worst fractal dimension', y='mean compactness', data=df, hue='benign_0__mal_1', ax=ax[2, 2])
plt.show()

# Train test split

X = df.drop('benign_0__mal_1', axis=1).values

y = df['benign_0__mal_1'].values

# Remember we need np.arrays

from sklearn.model_selection import train_test_split

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


