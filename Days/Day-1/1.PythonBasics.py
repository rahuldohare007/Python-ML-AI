# 🚀 Day 1/100 of #100DaysOfCode
# 🎯 Python Basics: Variables, Data Types & Operators

# Variable declarations with different data types--------------------------------------------------------------------------------------------------------

# Q. What is a variable?
# A. Variable is like a container that holds data. Very similar to how our containers in kitchen holds sugar, salt etc Creating a variable is like
#    creating a placeholder in memory and assigning it some value. In Python its as easy as writing:
name = "Rahul"  # str
age = 25  # int
height = 5.8  # float
is_coder = True  # bool

# Type checking-------------------------------------------------------------------------------------------------------------------------------------------

# Q. What is a Data Type?
# A. Data type specifies the type of value a variable holds. This is required in programming to do various operations without causing an error.
#    In python, we can print the type of any operator using type function:
print(type(name))  # <class 'str'>
print(type(age))  # <class 'int'>
print(type(height))  # <class 'float'>
print(type(is_coder))  # <class 'bool'>

# Operators behaving differently--------Output------------------------------------------------------------------------------------------------------------
print("Hello " + name)              # Hello Rahul      # + as string concatenation
print(age + 5)                      # 30               # + as numeric addition
print(name * 3)                     # RahulRahulRahul  # * repeats the string
print(age * 2)                      # 50               # * as numeric multiplication

