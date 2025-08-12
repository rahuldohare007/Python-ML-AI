# 🚀 Day 20/100 of #100DaysOfCode
# 🎯 Logistic Regression Mini Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import make_scorer, precision_score
scorer = make_scorer(precision_score, zero_division=0)


# Load dataset
df = pd.read_csv("../data/titanic.csv")  # Change path as needed

# --- EDA ---
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")

# Survival rate by sex
sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Sex")
plt.show()

# Age distribution
sns.histplot(df["Age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# --- Feature Engineering ---
# Map Sex
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
print(df.isnull().sum())

# Fill missing Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Create FamilySize feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Extract title from name
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

# Create dummy variables for Title
title_dummies = pd.get_dummies(df['Title'], prefix="Title", drop_first=True)
df = pd.concat([df, title_dummies], axis=1)

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin', 'Title'], axis=1, inplace=True)

# Get dummies for Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# --- Features & Target ---
X = df.drop('Survived', axis=1)
y = df['Survived']

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Model with Hyperparameter Tuning ---
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\nBest Parameters:", grid.best_params_)

# --- Predictions ---
y_pred = best_model.predict(X_test)

# --- Evaluation ---
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))


