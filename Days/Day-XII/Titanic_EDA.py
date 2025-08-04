# 🚀 Day 12/100 of #100DaysOfCode
# 🎯 EDA Continued + Handling Missing Values 


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('titanic')

# --- Handling Missing Values ---
# Check missing values
print("Missing values before:\n", df.isnull().sum())

# Fill 'age' with median
df['age'] = df['age'].fillna(df['age'].median())

# Fill 'embarked' with mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop rows where 'deck' is missing (too many nulls)
df = df.drop(columns=['deck'])

# Drop any remaining rows with missing target variable
df = df.dropna(subset=['survived'])

print("\nMissing values after:\n", df.isnull().sum())

# --- EDA Visualizations ---
sns.boxplot(x='class', y='age', data=df)
plt.title('Age Distribution by Class')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
