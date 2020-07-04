import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.keys())

print(cancer['DESCR'])

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

print(df)

from sklearn.preprocessing import StandardScaler

# We are going to scale our data

scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)

print(scaled_data.shape)
# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
print(x_pca.shape)

sns.set_style('whitegrid')
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], edgecolors='black', linewidths=0.2, c=cancer['target'], cmap='plasma')
plt.xlabel('PCA 1 First Principal Component')
plt.ylabel('PCA2 Second Principal Component')
plt.title('PCA Reduction')
plt.show()


# The principal components, PCA1 and PCA2 are a combination of all the features. We can show the correlation of each
# feature with the principal component. Remember from the book, tha PCA it's in reality a vector which their values
# are the % of the weight they have on the PCA

print(pca.components_)

heat_components = pd.DataFrame(pca.components_, columns=cancer.feature_names)

plt.figure(figsize=(14, 10))
sns.heatmap(heat_components, cmap='plasma')
plt.show()

# Now we could use some methods as SVM or logistic Regression to classify
