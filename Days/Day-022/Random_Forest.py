# 🚀 Day 22/100 of #100DaysOfCode
# 🎯 Random Forest - Bagging, Feature Importance

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# 2. Split data into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Train Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=100,       # number of trees
    criterion='gini',       # split criterion ('entropy' also works)
    max_depth=None,         # fully grown trees unless specified
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
rf.fit(X_train, y_train)

# 4. Evaluate model
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Feature importance visualization
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=iris.feature_names).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# 6. Check predictions vs actual
results = pd.DataFrame({'Actual': y_test.map(dict(enumerate(iris.target_names))),
                        'Predicted': pd.Series(y_pred).map(dict(enumerate(iris.target_names)))})
print("\nSample Predictions:\n", results.head(10))
