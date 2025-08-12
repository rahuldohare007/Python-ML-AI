# 🚀 Day 2/100 of #100DaysOfCode
# 🎯 Python Control Flow - if/else, loops

# Example 1: if-elif-else (conditional branching)
score = int(input("Enter your score (0–100): "))

if score >= 90:
    print("Grade: A – Excellent!")
elif score >= 75:
    print("Grade: B – Good job!")
elif score >= 60:
    print("Grade: C – Can improve")
else:
    print("Grade: F – Try harder next time!")

# Example 2: for loop – iterating over a list
fruits = ["Apple", "Banana", "Cherry"]
print("\nMy favorite fruits:")
for fruit in fruits:
    print(f" - I like {fruit}")

# Example 3: while loop – countdown logic
print("\nCountdown begins:")
count = 5
while count > 0:
    print(f"Countdown: {count}")
    count -= 1
print("Liftoff!")

# Bonus Example 4: using 'break' and 'continue'
print("\nFinding the first even number in a list:")
numbers = [1, 3, 7, 8, 11, 13]
for num in numbers:
    if num % 2 == 0:
        print(f"First even number found: {num}")
        break  # exits loop after finding first even number

print("\nSkipping odd numbers:")
for num in range(1, 6):
    if num % 2 != 0:
        continue  # skip the rest of the loop for odd numbers
    print(f"Even number: {num}")
