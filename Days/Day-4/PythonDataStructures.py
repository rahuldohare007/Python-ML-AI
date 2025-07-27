# 🚀 Day 4/100 of #100DaysOfCode
# 🎯 Python Data Structures

# === LISTS ===
# Mutable, ordered collection. Great for dynamic data.
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
print("List Example:", fruits)  # ['apple', 'banana', 'cherry', 'orange']
print("2nd fruit:", fruits[1])  # banana

# === TUPLES ===
# Immutable, ordered collection. Used for fixed data.
coordinates = (10.5, 20.3)
print("Tuple Example:", coordinates)  # (10.5, 20.3)
# coordinates[0] = 5  # This will raise an error

# === DICTIONARIES ===
# Key-value pairs. Great for fast lookup and structured data.
student = {
    "name": "Rahul",
    "age": 24,
    "course": "Python"
}
print("Dictionary Example:", student)  # {'name': 'Rahul', 'age': 24, 'course': 'Python'}
print("Student Name:", student["name"])  # Rahul

# Updating dictionary
student["age"] = 25
student["city"] = "Bhopal"
print("Updated Dictionary:", student)

# === SETS ===
# Unordered collection of unique items. Useful for membership tests and deduplication.
unique_numbers = {1, 2, 3, 3, 4, 2}
print("Set Example:", unique_numbers)  # {1, 2, 3, 4}

# Add and remove items
unique_numbers.add(5)
unique_numbers.discard(2)
print("Modified Set:", unique_numbers)  # e.g., {1, 3, 4, 5}

# === INTERSECTION OF STRUCTURES ===
# Convert list to set to remove duplicates
duplicates = ["a", "b", "a", "c", "b"]
no_duplicates = list(set(duplicates))
print("No Duplicates:", no_duplicates)  # ['a', 'c', 'b']

