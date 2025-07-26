# 🚀 Day 3/100 of #100DaysOfCode
# 🎯 Functions, Modules, and File I/O

import math  # Built-in module
import my_module  # Custom module (my_module.py must be in the same directory)


# 1. Functions
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


# 2. Using built-in module
root = math.sqrt(16)
print(f"Square root of 16 is: {root}")

# 3. Using custom module
print(my_module.greet("Rahul"))
print(f"Square of 7 is: {my_module.square(7)}")

# 4. File I/O
file_path = "data.txt"

# Write to file
with open(file_path, "w") as file:
    file.write("Learning File I/O in Python\n")
    file.write("It’s powerful and easy!\n")

# Read from file
with open(file_path, "r") as file:
    content = file.read()
    print("\n--- File Content ---")
    print(content)

# Note
# The f in f"Square root of 16 is: {root}" stands for f-string, short for formatted string literal —
# a Python feature introduced in Python 3.6.
