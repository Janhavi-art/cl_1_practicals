import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("wine.csv")

# Separate features and target
X = df.drop("Customer_Segment", axis=1)
y = df["Customer_Segment"]

# Standardize the data
X = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Display explained variance
print("Explained Variance:", pca.explained_variance_ratio_)

# Plot PCA results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Dataset")
plt.show()

# Principal Component Analysis (PCA) is a dimensionality reduction technique
# used to reduce the number of features while retaining maximum information.
# It transforms the original correlated features into new uncorrelated variables
# called Principal Components, which capture maximum variance in the data.
# Before applying PCA, Standardization is performed to bring all features
# to a common scale (mean = 0 and standard deviation = 1).
# The first principal component (PC1) captures the maximum variance, while
# the second principal component (PC2) captures the next highest variance
# and is perpendicular (orthogonal) to the first component.
# Explained Variance Ratio indicates the proportion of total information
# retained by each principal component.
# Finally, the data is visualized in a 2D scatter plot using PC1 and PC2,
# with different colours representing different customer segments.