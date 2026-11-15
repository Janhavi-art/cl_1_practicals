
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load dataset
df = pd.read_csv('uber.csv')

# --- TASK 1 & 2: Pre-process dataset & Identify/Remove Outliers ---
df = df[
    (df['fare_amount'] > 0)
    & (df['fare_amount'] < 200)
    & (df['pickup_latitude'].between(40, 42))
    & (df['dropoff_latitude'].between(40, 42))
    & (df['pickup_longitude'].between(-75, -73))
    & (df['dropoff_longitude'].between(-75, -73))
]

# Feature Engineering
df['distance'] = np.sqrt(
    (df['dropoff_latitude'] - df['pickup_latitude']) ** 2
    + (df['dropoff_longitude'] - df['pickup_longitude']) ** 2
)
df['hour'] = pd.to_datetime(df['pickup_datetime']).dt.hour
df['day_of_week'] = pd.to_datetime(df['pickup_datetime']).dt.dayofweek
# --- TASK 3: Check the correlation ---
cols_to_check = [
    'fare_amount',
    'distance',
    'passenger_count',
    'hour',
    'day_of_week',
]
correlation_matrix = df[cols_to_check].corr()
print('=== CORRELATION WITH FARE AMOUNT ===')
print(correlation_matrix['fare_amount'])
print('\n' + '=' * 40 + '\n')

# Define X and y
X = df[['distance', 'passenger_count', 'hour', 'day_of_week']].fillna(0)
y = df['fare_amount']

# Scaling and Train-Test Split
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# --- TASK 4 & 5: Implement Models, Evaluate, and Compare (R2, RMSE) ---
print('=== MODEL PERFORMANCE ===')
for model in [LinearRegression(), Ridge(), Lasso()]:
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  r2 = r2_score(y_test, pred)
  rmse = mean_squared_error(y_test, pred) ** 0.5
  print(f'{type(model).__name__:<18} | R2: {r2:.4f} | RMSE: {rmse:.4f}')

 