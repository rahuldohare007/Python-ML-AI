# 🚀 Day 14/100 of #100DaysOfCode
# 🎯 Project: Kaggle Dataset EDA (End-to-End)  

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("../data/titanic.csv")

# 1. Quick Overview
print(df.head())
print(df.info())
print(df.describe())

# 2. Handle Missing Values

# Fill 'age' with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill 'Embarked' with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop 'Cabin' due to too many missing values
df.drop(columns=['Cabin'], inplace=True)

# 3. Data Cleaning & Encoding

# Encode 'Sex' and 'Embarked'
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


# 4. Exploratory Data Analysis

# Survival by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

# Age distribution by survival
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age Distribution by Survival')
plt.show()

# Heatmap of correlations
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot: Age by Passenger Class
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age by Passenger Class')
plt.show()

# 5. Insights
print("\nSurvival Rate by Class and Gender:")
print(df.groupby(['Pclass', 'Sex'])['Survived'].mean())

print("\nAverage Fare by Embarkation Port:")
print(df.groupby('Embarked')['Fare'].mean())

# Save survival rate by class and gender to CSV
survival_rate = df.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index()
survival_rate.to_csv('../output/survival_rate_by_class.csv', index=False)

print("\nSaved survival rate by class and gender to: output/survival_rate_by_class.csv")