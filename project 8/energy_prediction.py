import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel('H:/AI/project 8/ENB2012_data.xlsx', engine='openpyxl')

df.columns = ['Compactness', 'SurfaceArea', 'WallArea', 'RoofArea',
              'Height', 'Orientation', 'GlazingArea', 'GlazingDist',
              'HeatingLoad', 'CoolingLoad']

print("=" * 50)
print("ENERGY DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal buildings : {df.shape[0]}")
print(f"Total features  : {df.shape[1] - 2}")
print(f"\n--- First 5 rows ---")
print(df.head())
print(f"\n--- Statistics ---")
print(df.describe())

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['HeatingLoad'].hist(bins=20, color='red', alpha=0.7)
plt.title('Heating Load Distribution')
plt.xlabel('Heating Load')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
df['CoolingLoad'].hist(bins=20, color='blue', alpha=0.7)
plt.title('Cooling Load Distribution')
plt.xlabel('Cooling Load')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
corr = df.corr()
sns.heatmap(corr[['HeatingLoad', 'CoolingLoad']].drop(['HeatingLoad', 'CoolingLoad']),
            annot=True, fmt='.2f', cmap='coolwarm', ax=plt.gca())
plt.title('Feature Correlations')

plt.tight_layout()
plt.savefig('H:/AI/project 8/exploration.png')
plt.show()

X = df.drop(['HeatingLoad', 'CoolingLoad'], axis=1)
y = df['HeatingLoad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set : {X_train.shape[0]} buildings")
print(f"Test set     : {X_test.shape[0]} buildings")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n--- Data normalized (important for SVR) ---")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)

print(f"\n--- Linear Regression ---")
print(f"RMSE : {lr_rmse:.2f}")
print(f"R2   : {lr_r2:.2f}")

svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
svr_pred = svr_model.predict(X_test_scaled)

svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
svr_r2   = r2_score(y_test, svr_pred)

print(f"\n--- SVR (Support Vector Regression) ---")
print(f"RMSE : {svr_rmse:.2f}")
print(f"R2   : {svr_r2:.2f}")

print(f"\n--- Model Comparison ---")
print(f"Linear Regression : RMSE={lr_rmse:.2f}  R2={lr_r2:.2f}")
print(f"SVR               : RMSE={svr_rmse:.2f}  R2={svr_r2:.2f}")
best = "SVR" if svr_r2 > lr_r2 else "Linear Regression"
print(f"Best model        : {best}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_pred, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Linear Regression\nR2={lr_r2:.2f}')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')

plt.subplot(1, 2, 2)
plt.scatter(y_test, svr_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'SVR\nR2={svr_r2:.2f}')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')

plt.tight_layout()
plt.savefig('H:/AI/project 8/predictions.png')
plt.show()

print(f"\n--- Predict a new building ---")
new_building = X_test.iloc[0:1]
new_building_scaled = scaler.transform(new_building)

lr_result  = lr_model.predict(new_building_scaled)[0]
svr_result = svr_model.predict(new_building_scaled)[0]
actual     = y_test.iloc[0]

print(f"Linear Regression predicts : {lr_result:.2f} kWh/m2")
print(f"SVR predicts               : {svr_result:.2f} kWh/m2")
print(f"Actual heating load        : {actual:.2f} kWh/m2")
