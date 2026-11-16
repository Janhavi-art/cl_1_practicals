
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

 # Linear Regression is a supervised machine learning algorithm used to predict
# continuous numerical values by establishing a linear relationship between
# independent variables (features) and a dependent variable (target).
#
# Data Preprocessing involves cleaning the dataset by removing invalid values
# and outliers to improve the accuracy and reliability of the model.
#
# Feature Engineering is the process of creating new meaningful features
# from existing data. Here, distance is calculated using latitude and
# longitude, while hour and day_of_week are extracted from pickup_datetime.
#
# Correlation measures the strength and direction of the relationship
# between two variables. The correlation coefficient ranges from -1 to +1.
# Positive correlation indicates both variables increase together, while
# negative correlation indicates one decreases as the other increases.
#
# Standardization transforms features to have a mean of 0 and standard
# deviation of 1, ensuring that features with larger scales do not dominate.
#
# Train-Test Split divides the dataset into training and testing sets.
# The training set is used to train the model, while the testing set
# evaluates its performance on unseen data.
#
# Linear Regression predicts values using a linear equation.
# Ridge Regression uses L2 regularization, which adds a penalty based on
# the square of coefficient values to reduce overfitting.
# Lasso Regression uses L1 regularization, which adds a penalty based on
# the absolute values of coefficients and can reduce some coefficients to zero.
#
# R-squared (R2) measures how well the model explains the variance in the
# target variable. A value closer to 1 indicates a better fit.
#
# Root Mean Squared Error (RMSE) measures the average prediction error
# by calculating the square root of the mean squared differences between
# actual and predicted values. Lower RMSE indicates better performance.
#
# Finally, Linear, Ridge, and Lasso Regression models are compared using
# R2 Score and RMSE to identify the model with better predictive performance.