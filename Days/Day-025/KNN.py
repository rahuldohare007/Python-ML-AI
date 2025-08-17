# 🚀 Day 25/100 of #100DaysOfCode
# 🎯 KNN + Model Selection (train_test_split) 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Test different k values to select the best model
k_values = range(1, 16)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k={k}: Accuracy = {acc:.3f}")

# 4. Plot accuracy vs k
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='--')
plt.title('KNN Accuracy for Different k Values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# 5. Select best k and evaluate
best_k = k_values[accuracies.index(max(accuracies))]
print(f"\nBest k: {best_k} with Accuracy = {max(accuracies):.3f}")

# Train final model
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
final_preds = best_knn.predict(X_test)

# Detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, final_preds, target_names=iris.target_names))
