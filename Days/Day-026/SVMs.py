# 🚀 Day 26/100 of #100DaysOfCode
# 🎯 SVMs (Linear & Non-linear) 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset (binary classification for visualization)
X, y = datasets.make_classification(
    n_samples=300, n_features=2, n_redundant=0, 
    n_informative=2, n_clusters_per_class=1, random_state=42
)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Train Linear SVM
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train, y_train)

# 3. Train Non-linear SVM (RBF Kernel)
rbf_svm = SVC(kernel='rbf', gamma=0.5, C=1.0)
rbf_svm.fit(X_train, y_train)

# 4. Evaluate both models
y_pred_linear = linear_svm.predict(X_test)
y_pred_rbf = rbf_svm.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))

# 5. Function to plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.show()

# 6. Visualize decision boundaries
plot_decision_boundary(linear_svm, X_train, y_train, "Linear SVM Decision Boundary")
plot_decision_boundary(rbf_svm, X_train, y_train, "RBF Kernel SVM Decision Boundary")
