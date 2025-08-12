# 🚀 Day 9/100 of #100DaysOfCode
# 🎯 Pandas GroupBy, Merge & Pivot Tables

import pandas as pd

# GroupBy Example
data = {
    'Department': ['Sales', 'Sales', 'HR', 'HR', 'IT', 'IT'],
    'Employee': ['Rahul', 'Raj', 'Ritika', 'Mansi', 'Luffy', 'Zoro'],
    'Salary': [50000, 55000, 40000, 42000, 60000, 62000]
}
df = pd.DataFrame(data)

grouped = df.groupby('Department')['Salary'].mean()
print("Average Salary by Department:\n", grouped)

#  Merge Example
left = pd.DataFrame({
    'EmpID': [1, 2, 3],
    'Name': ['Ram', 'Shyam', 'Harish']
})

right = pd.DataFrame({
    'EmpID': [1, 2, 3],
    'Score': [88, 92, 85]
})

merged = pd.merge(left, right, on='EmpID')
print("\nMerged DataFrame:\n", merged)

#  Pivot Table Example
sales_data = {
    'Region': ['North', 'North', 'South', 'South'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250]
}
df_sales = pd.DataFrame(sales_data)

pivot = pd.pivot_table(df_sales, values='Sales', index='Region', columns='Product')
print("\nPivot Table:\n", pivot)
