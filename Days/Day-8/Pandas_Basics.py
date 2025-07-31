# 🚀 Day 8/100 of #100DaysOfCode
# 🎯 Mastered Pandas Series & DataFrames  

import pandas as pd

# Series
data = [10, 20, 30, 40]
labels = ['a', 'b', 'c', 'd']
series = pd.Series(data, index=labels)
print("Series:\n", series)

# Accessing Series data
print("\nAccess 'b':", series['b'])


# DataFrame
data_dict = {
    'Name': ['Ram', 'Shyam', 'Harish'],
    'Age': [25, 30, 22],
    'City': ['Delhi', 'Mumbai', 'Bangalore']
}
df = pd.DataFrame(data_dict)
print("\nDataFrame:\n", df)


# Indexing & Filtering
print("\nRow 0:\n", df.loc[0])
print("\nAge > 24:\n", df[df['Age'] > 24])


# Reading CSV (sample.csv)

# sample.csv should be placed in the same folder
csv_df = pd.read_csv('sample.csv')
print("\nCSV Data:\n", csv_df)

# Basic Analysis
print("\nDescribe:\n", csv_df.describe())
print("Mean Score:", csv_df['Score'].mean())
