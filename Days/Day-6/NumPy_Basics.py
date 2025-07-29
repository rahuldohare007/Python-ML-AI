# 🚀 Day 6/100 of #100DaysOfCode
# 🎯 Dived into NumPy basics

import numpy as np

# 1. Creating NumPy Arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print("1D Array:", arr1)
print("2D Array:\n", arr2)

# 2. Array Properties
print("\nShape of arr2:", arr2.shape)
print("Data type:", arr2.dtype)
print("Dimensions:", arr2.ndim)
print("Size (total elements):", arr2.size)


# 3. Indexing and Slicing
print("\nElement at arr1[1]:", arr1[1])         # 2
print("Element at arr2[0,2]:", arr2[0, 2])      # 3
print("Slicing arr1[0:2]:", arr1[0:2])          # [1 2]

# 4. Broadcasting
broadcasted = arr1 + 5
print("\nBroadcasted (arr1 + 5):", broadcasted)  # [6 7 8]

# Add 1D array to 2D array (row-wise)
broadcasted_2d = arr2 + arr1
print("Broadcasted 2D (arr2 + arr1):\n", broadcasted_2d)

# 5. Element-wise Operations
print("\nMultiplying arrays:", arr1 * 2)        # [2 4 6]
print("Square of arr1:", arr1 ** 2)             # [1 4 9]

# 6. Aggregation Functions
print("\nSum of arr2:", np.sum(arr2))
print("Max of arr2:", np.max(arr2))
print("Min of arr2:", np.min(arr2))
print("Mean of arr2:", np.mean(arr2))
print("Standard Deviation:", np.std(arr2))

# 7. Reshaping Arrays
arr3 = np.arange(12)            # [0, 1, ..., 11]
reshaped = arr3.reshape(3, 4)   # 3 rows, 4 cols
print("\nReshaped 3x4 Array:\n", reshaped)

# 8. Stacking Arrays
a = np.array([1, 2])
b = np.array([3, 4])
print("\nHorizontal Stack:", np.hstack([a, b]))  # [1 2 3 4]
print("Vertical Stack:\n", np.vstack([a, b]))    # [[1 2], [3 4]]
