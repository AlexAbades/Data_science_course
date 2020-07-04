import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

# We create a random data with make blobs, n=200, p=2
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std= 1.8, random_state=101)

print(data)

# The second item on the data, data[1] it refers to the cluster that it belong

# We have to grab the first element of the data "0" to call the shape function
print(data[0].shape)

# the features are in data[0]

# We are selecting the features, [0] and then, we are selecting all the rows of the first column. They are tuples
print(data[0][:, 0])
sns.set_style('whitegrid')
plt.scatter(data[0][:,0], data[0][:, 1], c=data[1], cmap='rainbow', edgecolors='black', linewidths=0.2)
plt.show()

from sklearn.cluster import KMeans

# We have to specify the numbers of clusters
kmeans = KMeans(n_clusters=4)

kmeans.fit(data[0])
# We can ask for the cluster centers
print(kmeans.cluster_centers_)

# or we can ask for the labels it believes to be tru for the cluster
print(kmeans.labels_)

# We can plot the two data, the true values with our predicted
# Sharey, to share the same axes

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))

ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')
plt.show()