# 🚀 Day 15/100 of #100DaysOfCode
# 🎯 Recap: Python for Data Science — Quiz + Notes

import pandas as pd

# Sample employee dataset
data = {
    "Name": ["Rahul", "Ritik", "Ravi", "Raj", "Ram"],
    "Department": ["HR", "IT", "Finance", "IT", "HR"],
    "Salary": [55000, 68000, 72000, 60000, 52000],
    "Experience": [2, 5, 8, 3, 1],
}

df = pd.DataFrame(data)

# 1. Indexing
print(" First 3 rows:\n", df.head(3))

# 2. Filtering: Employees with Salary > 60000
high_salary = df[df["Salary"] > 60000]
print("\n Employees with Salary > 60000:\n", high_salary)

# 3. Aggregation: Average salary by department
avg_salary_dept = df.groupby("Department")["Salary"].mean()
print("\n Average Salary by Department:\n", avg_salary_dept)

# 4. Advanced Filtering: IT Department with >3 years experience
experienced_it = df[(df["Department"] == "IT") & (df["Experience"] > 3)]
print("\n IT Employees with >3 Years Experience:\n", experienced_it)
