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