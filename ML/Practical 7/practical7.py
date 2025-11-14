import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import math


url = "uber.csv"
df = pd.read_csv(url)

print("Dataset Loaded Successfully ")
print(df.head())
print(df.info())


df = df.dropna()

if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['year'] = df['pickup_datetime'].dt.year

df = df.drop('pickup_datetime', axis=1)

print("\nAfter preprocessing:")
print(df.head())


plt.figure(figsize=(6,4))
sns.boxplot(x=df['fare_amount'])
plt.title("Outliers in Fare Amount")
plt.show()

df = df[(df['fare_amount'] > 0) & (df['fare_amount'] <= 100)]


numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap (Numeric Columns Only)")
plt.show()


X = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
        'dropoff_latitude', 'passenger_count', 'hour', 'day', 'month', 'year']]
y = df['fare_amount']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData Split Successful")



lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)



rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = math.sqrt(mean_squared_error(y_test, y_pred_lr))

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = math.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\nModel Evaluation Results ")
print("Linear Regression:")
print("  R² Score:", round(r2_lr, 3))
print("  RMSE:", round(rmse_lr, 3))

print("\nRandom Forest Regressor:")
print("  R² Score:", round(r2_rf, 3))
print("  RMSE:", round(rmse_rf, 3))


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R2 Score': [r2_lr, r2_rf],
    'RMSE': [rmse_lr, rmse_rf]
})
print("\nModel Comparison:\n", results)