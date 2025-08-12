# 🚀 Day 18/100 of #100DaysOfCode
# 🎯 Linear Regression Project (House Price)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
df = pd.read_csv("../data/house_prices.csv")

# 2. Inspect data
print("Shape:", df.shape)
print(df.head())

# 3. Select relevant features based on correlation with target
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF"]
X = df[features]
y = df["SalePrice"]

# Handle missing values (if any)
X = X.fillna(0)
y = y.fillna(y.median())

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model training
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root MSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8. Example: Predict for a new house
sample_data = pd.DataFrame(
    {"OverallQual": [7], "GrLivArea": [2000], "GarageCars": [2], "TotalBsmtSF": [900]}
)
predicted_price = model.predict(sample_data)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
