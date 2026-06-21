import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('H:/AI/project 10/weatherHistory.csv')

print("=" * 50)
print("WEATHER DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal records : {df.shape[0]}")
print(f"Total features: {df.shape[1]}")
print(f"\n--- First 5 rows ---")
print(df[['Formatted Date', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']].head())

df = df.dropna()
print(f"\nAfter removing missing values: {df.shape[0]} rows")

df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['hour']  = df['Formatted Date'].dt.hour
df['month'] = df['Formatted Date'].dt.month
df['day']   = df['Formatted Date'].dt.day

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['Temperature (C)'].hist(bins=30, color='steelblue')
plt.title('Temperature Distribution')
plt.xlabel('Temperature (C)')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
sample = df.iloc[:720]
plt.plot(sample['Temperature (C)'].values, color='orange', linewidth=0.8)
plt.title('Temperature - First 30 Days')
plt.xlabel('Hour')
plt.ylabel('Temperature (C)')

plt.subplot(1, 3, 3)
plt.scatter(df['Humidity'], df['Temperature (C)'], alpha=0.1, color='green', s=1)
plt.title('Humidity vs Temperature')
plt.xlabel('Humidity')
plt.ylabel('Temperature (C)')

plt.tight_layout()
plt.savefig('H:/AI/project 10/exploration.png')
plt.show()

features = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
            'Visibility (km)', 'Pressure (millibars)', 'hour', 'month', 'day']

X = df[features]
y = df['Temperature (C)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set : {X_train.shape[0]} records")
print(f"Test set     : {X_test.shape[0]} records")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n--- Training Linear Regression... ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)
print(f"RMSE : {lr_rmse:.2f}°C")
print(f"R2   : {lr_r2:.2f}")

print("\n--- Training Random Forest... ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)
print(f"RMSE : {rf_rmse:.2f}°C")
print(f"R2   : {rf_r2:.2f}")

print(f"\n--- Model Comparison ---")
print(f"Linear Regression : RMSE={lr_rmse:.2f}  R2={lr_r2:.2f}")
print(f"Random Forest     : RMSE={rf_rmse:.2f}  R2={rf_r2:.2f}")
best = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
print(f"Best model        : {best}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_pred, alpha=0.1, color='steelblue', s=2)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Linear Regression\nR2={lr_r2:.2f}')
plt.xlabel('Actual Temperature (C)')
plt.ylabel('Predicted Temperature (C)')

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_pred, alpha=0.1, color='green', s=2)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Random Forest\nR2={rf_r2:.2f}')
plt.xlabel('Actual Temperature (C)')
plt.ylabel('Predicted Temperature (C)')

plt.tight_layout()
plt.savefig('H:/AI/project 10/predictions.png')
plt.show()

importances = pd.Series(rf_model.feature_importances_, index=features)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(8, 4))
importances.plot(kind='bar', color='steelblue')
plt.title('Feature Importance for Temperature Prediction')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('H:/AI/project 10/feature_importance.png')
plt.show()

print(f"\n--- Top features for predicting temperature ---")
print(importances.to_string())

print(f"\n--- Predict temperature for a new record ---")
new_record = X_test.iloc[0:1]
new_scaled = scaler.transform(new_record)
lr_temp = lr_model.predict(new_scaled)[0]
rf_temp = rf_model.predict(new_scaled)[0]
actual  = y_test.iloc[0]

print(f"Linear Regression predicts : {lr_temp:.1f}°C")
print(f"Random Forest predicts     : {rf_temp:.1f}°C")
print(f"Actual temperature         : {actual:.1f}°C")
