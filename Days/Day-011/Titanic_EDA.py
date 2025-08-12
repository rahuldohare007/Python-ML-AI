# 🚀 Day 11/100 of #100DaysOfCode
# 🎯 Exploratory Data Analysis (EDA) on Titanic Dataset

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Drop rows with missing values in key columns
df = df.dropna(subset=['age', 'sex', 'class', 'survived'])

# Pairplot to visualize relationships
sns.pairplot(df[['age', 'fare', 'survived']], hue='survived')
plt.suptitle("Pairplot: Age, Fare & Survival", y=1.02)
plt.show()

# Countplot for survival by gender
sns.countplot(data=df, x='sex', hue='survived')
plt.title('Survival Count by Gender')
plt.show()

# Grouped bar chart: survival by class and gender
survival_rate = df.groupby(['class', 'sex'], observed=True)['survived'].mean().unstack()
survival_rate.plot(kind='bar')
plt.title('Survival Rate by Class & Gender')
plt.ylabel('Survival Rate')
plt.show()