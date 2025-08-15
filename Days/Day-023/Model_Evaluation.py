# 🚀 Day 23/100 of #100DaysOfCode
# 🎯 Model Evaluation (Accuracy, Confusion Matrix)

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Make classes imbalanced by removing most of class 2
X = X[y != 2]
y = y[y != 2]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Logistic Regression model
clf = LogisticRegression(max_iter=200, solver='lbfgs')
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# --- Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# --- Classification Report ---
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names[:2]))

# --- Visualization of Confusion Matrix ---
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names[:2], 
            yticklabels=iris.target_names[:2])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()
