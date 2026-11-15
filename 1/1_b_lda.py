import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("iris.csv")

# Separate features and target
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Predict species
y_pred = lda.predict(X_test)

# Display accuracy
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Predict a new flower
flower = [[5.1, 3.5, 1.4, 0.2]]
print("Predicted Species:", lda.predict(flower)[0])