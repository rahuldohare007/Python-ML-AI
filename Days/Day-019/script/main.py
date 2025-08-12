# 🚀 Day 19/100 of #100DaysOfCode
# 🎯 Logistic Regression + Classification Metrics

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("../data/titanic.csv")

# Select features & target
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]
df = df[features + ["Survived"]]

# Encode categorical
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Split features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall:", round(recall_score(y_test, y_pred), 3))
print("F1-score:", round(f1_score(y_test, y_pred), 3))
