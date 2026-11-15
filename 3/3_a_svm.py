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