# 🚀 Day 21/100 of #100DaysOfCode
# 🎯 Decision Trees – Intuition + Practice

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Decision Tree classifier
clf = DecisionTreeClassifier(
    criterion='entropy',  # 'gini' also works
    max_depth=3,          # limit depth to avoid overfitting
    random_state=42
)
clf.fit(X_train, y_train)

# 4. Predict on test set
y_pred = clf.predict(X_test)

# 5. Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Visualize Decision Tree
plt.figure(figsize=(14, 10))
plot_tree(
    clf,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True,
    proportion=False,
    precision=2
)
plt.title("Decision Tree Classifier on Iris Dataset")
plt.show()
