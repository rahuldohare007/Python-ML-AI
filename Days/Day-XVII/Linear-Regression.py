# 🚀 Day 17/100 of #100DaysOfCode
# 🎯 Linear Regression — Theory + Implementation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (house price example)
data = {
    "area_sqft": [850, 900, 1000, 1200, 1500, 1800],
    "price": [150000, 165000, 185000, 210000, 250000, 300000]
}
df = pd.DataFrame(data)

# Features & target
X = df[['area_sqft']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Slope:", model.coef_[0], "Intercept:", model.intercept_)
