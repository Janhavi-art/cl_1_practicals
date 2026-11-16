import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
digits = datasets.load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, pred))
print(metrics.classification_report(y_test, pred))

# Support Vector Machine (SVM) is a supervised machine learning algorithm
# used for classification and regression tasks. It finds an optimal
# decision boundary (hyperplane) that separates different classes.
#
# The main objective of SVM is to maximize the margin, which is the distance
# between the hyperplane and the nearest data points called support vectors.
#
# The Digits dataset contains handwritten digit images (0–9), where each
# image is represented by 64 features corresponding to an 8x8 pixel image.
#
# Train-Test Split divides the dataset into training and testing sets.
# The training data is used to train the SVM model, while testing data
# evaluates its performance on unseen samples.
#
# Linear Kernel is used in SVM to create a linear decision boundary
# (hyperplane) between classes.
#
# Model Training involves learning the optimal hyperplanes using
# the training data.
#
# Prediction is performed on the test dataset to classify handwritten
# digit images into their respective numerical classes (0–9).
#
# Accuracy measures the proportion of correctly classified samples
# out of the total number of test samples.
#
# Confusion Matrix is a table used to evaluate classification performance.
# It compares actual class labels with predicted class labels and shows
# correct and incorrect classifications for each class.
#
# Classification Report provides precision, recall, F1-score and support.
# Precision measures the correctness of positive predictions.
# Recall measures how many actual positive samples are correctly identified.
# F1-score is the harmonic mean of precision and recall.
# Support represents the number of actual samples belonging to each class.