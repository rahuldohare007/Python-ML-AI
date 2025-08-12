# 🚀 Day 10/100 of #100DaysOfCode
# 🎯 Visualized data using Matplotlib & Seaborn 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [23, 45, 56, 78]
}
df = pd.DataFrame(data)

# Line Plot
plt.plot(df['Category'], df['Values'], marker='o')
plt.title('Line Plot Example')
plt.xlabel('Category')
plt.ylabel('Values')
plt.grid(True)
plt.show()

# Bar Plot
plt.bar(df['Category'], df['Values'], color='skyblue')
plt.title('Bar Plot Example')
plt.xlabel('Category')
plt.ylabel('Values')
plt.show()

# Histogram
np.random.seed(0)
rand_data = np.random.randn(100)
plt.hist(rand_data, bins=10, color='green', edgecolor='black')
plt.title('Histogram Example')
plt.show()

# Boxplot using Seaborn
sns.boxplot(data=rand_data, orient='h', color='orange')
plt.title('Boxplot Example')
plt.show()

# Heatmap using Seaborn
matrix = np.random.rand(4, 4)
sns.heatmap(matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap Example')
plt.show()
                                                                   