import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Import the dataset
df = pd.read_csv("telcom_customer_churn.csv")

# 2. Explore the dataset
print("--- Dataset Shape ---")
print(df.shape)
print("\n--- Dataset Info ---")
df.info()

# 6. Data types (done early because TotalCharges often has blank text like " ")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks become NaN
print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

# 3. Handle missing values: median for numbers, most common value for text
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(exclude="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# 4. Remove duplicate records
print(f"\nDuplicate rows found: {df.duplicated().sum()}")
df = df.drop_duplicates()

# 5. Check for inconsistent data: remove spaces, make lowercase, merge similar labels
text_cols = df.select_dtypes(exclude="number").columns.drop("customerID", errors="ignore")
for col in text_cols:
    df[col] = df[col].str.strip().str.lower()

df = df.replace({"no internet service": "no", "no phone service": "no"})

# 6. Data types (continued): Convert Churn yes/no -> 1/0
df["Churn"] = df["Churn"].map({"yes": 1, "no": 0})

# 7. Identify and handle outliers: clip extreme values using the IQR rule
print("\n--- Outlier Detection ---")
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    outliers_count = ((df[col] < low) | (df[col] > high)).sum()
    print(f"{col} outliers clipped: {outliers_count}")
    df[col] = df[col].clip(low, high)

# 8. Perform feature engineering
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)  # +1 avoids dividing by 0
df["TenureGroup"] = pd.cut(
    df["tenure"], 
    bins=[-1, 12, 24, 48, np.inf],
    labels=["0-1yr", "1-2yr", "2-4yr", "4yr+"]
)

# 11. Export the cleaned dataset for future analysis
df.to_csv("Telecom_Customer_Churn_cleaned.csv", index=False)
print("\nCleaned dataset exported to 'Telecom_Customer_Churn_cleaned.csv'")

# --- Prepare for modeling: turn text columns into 0/1 columns ---
X = pd.get_dummies(df.drop(columns=["Churn", "customerID"], errors="ignore"), drop_first=True)
y = df["Churn"]

# 10. Split the dataset (done BEFORE scaling so test data doesn't leak into the scaler)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Normalize or scale the data (fit on train only)
scale_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

print("\n--- Final Modeling Data Shapes ---")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Telecom Customer Churn Data Preprocessing:
# Data Preprocessing is the process of converting raw data into a clean,
# consistent and suitable format for machine learning.
#
# 1. Data Loading: Pandas read_csv() is used to load the Telecom Customer
# Churn dataset into a DataFrame.
#
# 2. Data Exploration: shape displays the number of rows and columns,
# while info() provides column names, data types and non-null counts.
# This helps understand the structure and quality of the dataset.
#
# 3. Data Type Conversion: pd.to_numeric() converts TotalCharges
# from object/string to numerical format. errors='coerce' converts
# invalid or blank values into NaN (Not a Number).
#
# 4. Missing Value Handling: Missing numerical values are replaced
# using the median, while missing categorical values are replaced
# using the mode (most frequently occurring value).
#
# 5. Duplicate Removal: duplicated() identifies repeated records,
# while drop_duplicates() removes duplicate rows to maintain data quality.
#
# 6. Inconsistent Data Handling: strip() removes leading and trailing
# spaces, lower() converts text into lowercase, and replace() merges
# similar categorical labels into a common category.
#
# 7. Categorical Encoding: The Churn column is converted into binary
# numerical values using mapping: Yes = 1 and No = 0.
#
# 8. Outlier Detection and Handling: The Interquartile Range (IQR)
# method is used to identify extreme values in numerical columns.
# Q1 represents the 25th percentile and Q3 represents the 75th percentile.
# IQR = Q3 - Q1.
# Lower Bound = Q1 - 1.5 * IQR.
# Upper Bound = Q3 + 1.5 * IQR.
# clip() limits values within these boundaries instead of removing rows.
#
# 9. Feature Engineering: Feature engineering creates new meaningful
# features from existing data to improve analysis and model performance.
# AvgMonthlySpend is calculated using TotalCharges and tenure.
# TenureGroup categorizes customers into different tenure intervals
# using pd.cut() and specified bin ranges.
#
# 10. Data Export: to_csv() saves the cleaned dataset into a new CSV file
# for future analysis and machine learning applications.
#
# 11. Categorical Feature Encoding: pd.get_dummies() performs One-Hot
# Encoding by converting categorical variables into binary columns.
# drop_first=True removes one category to avoid redundant columns.
# Churn is selected as the target variable (y), while the remaining
# customer attributes are selected as independent features (X).
#
# 12. Train-Test Split: The dataset is divided into 80% training data
# and 20% testing data using train_test_split().
# stratify=y maintains the same proportion of churn and non-churn
# customers in both training and testing datasets.
#
# 13. Feature Scaling: StandardScaler standardizes numerical features
# to have approximately mean 0 and standard deviation 1.
# fit_transform() calculates scaling parameters from training data
# and transforms it, while transform() applies the same parameters
# to testing data.
# This prevents data leakage from the testing dataset into training.
#
# 14. Final Data Verification: The shapes of X_train, X_test, y_train
# and y_test are displayed to verify the prepared modeling datasets.
#
# Finally, the cleaned and transformed dataset is ready for
# machine learning classification models to predict customer churn.