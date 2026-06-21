import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

print("=" * 50)
print("HOUSE PRICE DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal houses  : {df.shape[0]}")
print(f"Total features: {df.shape[1] - 1}")
print(f"\n--- First 5 rows ---")
print(df.head())
print(f"\n--- Statistics ---")
print(df.describe())

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['Price'].hist(bins=30, color='steelblue')
plt.title('House Price Distribution')
plt.xlabel('Price ($100,000s)')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
plt.scatter(df['AveRooms'], df['Price'], alpha=0.3, color='steelblue')
plt.title('Avg Rooms vs Price')
plt.xlabel('Average Rooms')
plt.ylabel('Price ($100,000s)')

plt.subplot(1, 3, 3)
plt.scatter(df['MedInc'], df['Price'], alpha=0.3, color='green')
plt.title('Median Income vs Price')
plt.xlabel('Median Income')
plt.ylabel('Price ($100,000s)')

plt.tight_layout()
plt.savefig('H:/AI/project 6/exploration.png')
plt.show()

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set : {X_train.shape[0]} houses")
print(f"Test set     : {X_test.shape[0]} houses")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)

print(f"\n--- Linear Regression ---")
print(f"RMSE : {lr_rmse:.2f} ($100,000s)")
print(f"R²   : {lr_r2:.2f}  (1.0 = perfect)")

dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2   = r2_score(y_test, dt_pred)

print(f"\n--- Decision Tree ---")
print(f"RMSE : {dt_rmse:.2f} ($100,000s)")
print(f"R²   : {dt_r2:.2f}  (1.0 = perfect)")

print(f"\n--- Model Comparison ---")
print(f"Linear Regression : RMSE={lr_rmse:.2f}  R²={lr_r2:.2f}")
print(f"Decision Tree     : RMSE={dt_rmse:.2f}  R²={dt_r2:.2f}")
best = "Decision Tree" if dt_r2 > lr_r2 else "Linear Regression"
print(f"Best model        : {best}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_pred, alpha=0.3, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Linear Regression\nR²={lr_r2:.2f}')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

plt.subplot(1, 2, 2)
plt.scatter(y_test, dt_pred, alpha=0.3, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Decision Tree\nR²={dt_r2:.2f}')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

plt.tight_layout()
plt.savefig('H:/AI/project 6/predictions.png')
plt.show()

print(f"\n--- Predict a new house ---")
new_house = X_test.iloc[0:1]
lr_price  = lr_model.predict(new_house)[0]
dt_price  = dt_model.predict(new_house)[0]
actual    = y_test.iloc[0]

print(f"Linear Regression predicts : ${lr_price * 100000:.0f}")
print(f"Decision Tree predicts     : ${dt_price * 100000:.0f}")
print(f"Actual price               : ${actual * 100000:.0f}")
