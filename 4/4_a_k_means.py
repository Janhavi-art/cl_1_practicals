from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
iris = load_iris(as_frame=True)
X = iris.data  # Features (Sepal length/width, Petal length/width)
y = iris.target  # True species labels (optional, for reference)

X = StandardScaler().fit_transform(X)
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    inertia.append(model.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('K-Means Clustering')
plt.show()

# K-Means Clustering is an unsupervised machine learning algorithm used
# to divide data into K distinct clusters based on similarity.
#
# Clustering groups data points such that points within the same cluster
# are more similar to each other than to points in other clusters.
#
# The Iris dataset contains 150 flower samples belonging to three species:
# Setosa, Versicolor and Virginica. Each sample has four features:
# sepal length, sepal width, petal length and petal width.
#
# Standardization is performed to bring all features to a common scale
# with mean 0 and standard deviation 1, preventing features with larger
# numerical ranges from dominating the clustering process.
#
# K-Means works by selecting K initial centroids, assigning each data point
# to its nearest centroid and recalculating centroids as the mean of
# assigned points. This process repeats until convergence.
#
# Inertia is the sum of squared distances between each data point and
# its assigned cluster centroid. Lower inertia indicates more compact clusters.
#
# The Elbow Method is used to determine the optimal number of clusters (K).
# It plots the number of clusters against inertia. The optimal K is generally
# selected at the elbow point, where the decrease in inertia starts slowing down.
#
# After selecting K = 3, the K-Means algorithm is applied to group the
# Iris dataset into three clusters.
#
# The fit_predict() method trains the model and returns cluster labels
# assigned to each data point.
#
# Finally, a scatter plot is used to visualize the clusters using the
# first two standardized features, with different colours representing
# different cluster assignments.