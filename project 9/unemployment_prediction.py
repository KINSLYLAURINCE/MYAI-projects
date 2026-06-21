import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df_raw = pd.read_csv('H:/AI/project 9/OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC,1.0+.UNE_LF_M...Y._T.Y_GE15..M.csv')

df = df_raw[['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']].copy()
df.columns = ['country', 'date', 'unemployment_rate']

print("=" * 50)
print("UNEMPLOYMENT DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal records : {df.shape[0]}")
print(f"Countries     : {df['country'].nunique()}")
print(f"Date range    : {df['date'].min()} to {df['date'].max()}")
print(f"\n--- Latest unemployment rates ---")
print(df.sort_values('unemployment_rate', ascending=False)[['country', 'date', 'unemployment_rate']].head(10).to_string(index=False))

plt.figure(figsize=(12, 5))
latest = df.sort_values('unemployment_rate', ascending=False)
plt.barh(latest['country'], latest['unemployment_rate'], color='steelblue')
plt.title('Unemployment Rate by Country (Latest)')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('H:/AI/project 9/unemployment_by_country.png')
plt.show()

df['year']  = df['date'].str[:4].astype(int)
df['month'] = df['date'].str[5:7].astype(int)
df['country_code'] = pd.Categorical(df['country']).codes

X = df[['country_code', 'year', 'month']]
y = df['unemployment_rate']

print(f"\n--- Features used ---")
print(f"country_code, year, month")
print(f"Total samples: {len(X)}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set : {X_train.shape[0]} records")
print(f"Test set     : {X_test.shape[0]} records")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)

print(f"\n--- Linear Regression ---")
print(f"RMSE : {lr_rmse:.2f}%")
print(f"R2   : {lr_r2:.2f}")

try:
    from xgboost import XGBRegressor
    xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2   = r2_score(y_test, xgb_pred)
    print(f"\n--- XGBoost ---")
    print(f"RMSE : {xgb_rmse:.2f}%")
    print(f"R2   : {xgb_r2:.2f}")
    best_pred = xgb_pred if xgb_r2 > lr_r2 else lr_pred
    best_name = "XGBoost" if xgb_r2 > lr_r2 else "Linear Regression"
    print(f"\n--- Model Comparison ---")
    print(f"Linear Regression : RMSE={lr_rmse:.2f}  R2={lr_r2:.2f}")
    print(f"XGBoost           : RMSE={xgb_rmse:.2f}  R2={xgb_r2:.2f}")
    print(f"Best model        : {best_name}")
except ImportError:
    print("\nXGBoost not installed. Install with: pip install xgboost")
    best_pred = lr_pred
    best_name = "Linear Regression"

plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_pred, alpha=0.7, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'{best_name} — Predicted vs Actual')
plt.xlabel('Actual Unemployment Rate (%)')
plt.ylabel('Predicted Unemployment Rate (%)')
plt.tight_layout()
plt.savefig('H:/AI/project 9/predictions.png')
plt.show()

print(f"\n--- Sample Predictions ---")
results = pd.DataFrame({
    'country': df.iloc[y_test.index]['country'].values,
    'actual':  y_test.values,
    'predicted': best_pred.round(2)
})
print(results.head(10).to_string(index=False))
