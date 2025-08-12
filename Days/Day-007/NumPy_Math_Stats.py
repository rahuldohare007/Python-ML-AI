# 🚀 Day 7/100 of #100DaysOfCode
# 🎯 Mastered NumPy Math Ops & Stats  

import numpy as np

# Sample data array
data = np.array([[10, 20, 30], [40, 50, 60]])

print("Data:\n", data)

# Basic Math Operations
print("\nAddition (each +5):\n", data + 5)
print("Multiplication (*2):\n", data * 2)
print("Square Root:\n", np.sqrt(data))
print("Power of 2:\n", np.power(data, 2))

#  Aggregate Functions
print("\nSum:", np.sum(data))
print("Row-wise Sum:", np.sum(data, axis=1))
print("Column-wise Mean:", np.mean(data, axis=0))

print("Max:", np.max(data))
print("Min:", np.min(data))
print("Range (ptp):", np.ptp(data))  # peak-to-peak

# Statistical Metrics
print("\nMean:", np.mean(data))
print("Median:", np.median(data))
print("Standard Deviation:", np.std(data))
print("Variance:", np.var(data))

# Correlation Coefficient
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
print("\nCorrelation Coefficient:\n", np.corrcoef(arr1, arr2))

# Cumulative Functions
print("\nCumulative Sum:\n", np.cumsum(data))
print("Cumulative Product:\n", np.cumprod(data))
