# 🚀 Day 16/100 of #100DaysOfCode
# 🎯 ML Pipeline: Titanic Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Sample dataset
df = pd.read_csv("../data/titanic.csv")

# Simple preprocessing: Drop NaN and encode gender
df = df.dropna(subset=['Age', 'Fare', 'Survived'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Age', 'Fare', 'Sex']]
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: Scaling + Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))
