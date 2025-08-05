# 🚀 Day 13/100 of #100DaysOfCode
# 🎯 SQL with Python – SELECT & JOIN (SQLite) 

import sqlite3
import pandas as pd

# Create in-memory SQLite DB
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create mock tables
cursor.execute('''
CREATE TABLE passengers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    class TEXT
)
''')

cursor.execute('''
CREATE TABLE survival (
    passenger_id INTEGER,
    survived INTEGER,
    FOREIGN KEY (passenger_id) REFERENCES passengers(id)
)
''')

# Insert sample data
cursor.executemany('INSERT INTO passengers VALUES (?, ?, ?, ?)', [
    (1, 'Rahul', 25, 'First'),
    (2, 'Ritik', 30, 'Third'),
    (3, 'Ravi', 22, 'Second')
])

cursor.executemany('INSERT INTO survival VALUES (?, ?)', [
    (1, 1),
    (2, 0),
    (3, 1)
])

conn.commit()

# JOIN query using Pandas
query = '''
SELECT p.name, p.age, p.class, s.survived
FROM passengers p
JOIN survival s ON p.id = s.passenger_id
'''

df = pd.read_sql_query(query, conn)
print(df)

# Close connection
conn.close()
