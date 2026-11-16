import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv('car_evaluation.csv')
data = data.apply(LabelEncoder().fit_transform)
X, y = data.iloc[:, :-1], data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Random Forest is a supervised machine learning algorithm used for
# classification and regression. It is an ensemble learning technique
# that combines multiple Decision Trees to improve prediction accuracy.
#
# The Car Evaluation dataset contains categorical features such as
# buying price, maintenance cost, number of doors, passenger capacity,
# luggage boot size and safety. The target represents car acceptability.
#
# Label Encoding converts categorical data into numerical values because
# machine learning algorithms generally require numerical input.
# Each unique category is assigned a numerical label.
#
# Feature Selection separates independent variables (X), which represent
# input features, and the dependent variable (y), which represents the
# target class to be predicted.
#
# Train-Test Split divides the dataset into training and testing sets.
# The training set is used to train the model, while the testing set
# evaluates its performance on unseen data.
#
# Random Forest creates multiple Decision Trees using different random
# samples of training data and feature subsets.
# Each Decision Tree makes an individual prediction, and the final
# classification is determined through majority voting.
#
# n_estimators specifies the number of Decision Trees in the forest.
# Here, 100 Decision Trees are used for classification.
#
# Model Training involves fitting the Random Forest classifier
# using the training dataset.
#
# Prediction is performed on the test dataset to classify the
# acceptability of cars.
#
# Accuracy measures the proportion of correctly classified samples
# out of the total test samples.
#
# Confusion Matrix compares actual and predicted class labels
# to identify correct and incorrect classifications.
#
# Classification Report provides precision, recall, F1-score and support
# for evaluating the performance of each class.