import os
import pandas as pd

# 1. Import and clean column names
df = pd.read_csv("bengaluru_house_data.csv")

print("========== BEFORE WRANGLING ==========")
print(f"Initial Shape: {df.shape}")
print("\nInitial Missing Values:")
print(df.isnull().sum())
print("\nInitial Data (first 3 rows):")
print(df.head(3))

df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

# 2. Missing values & Data formatting
df = df.drop(columns=["society", "availability"])
df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce") # forces ranges to NaN
df = df.dropna(subset=["location", "size", "total_sqft"])
df[["bath", "balcony"]] = df[["bath", "balcony"]].fillna(df[["bath", "balcony"]].median())

df["location"] = df["location"].str.strip().str.lower()
df["bhk"] = df["size"].str.split().str[0].astype(int)

print("==== AFTER HANDLING MISSING VALUES ===")
print(f"Missing values remaining: {df.isnull().sum().sum()}")
print(f"Rows remaining after drops: {len(df)}")

# 3. Merge (Optional)
if os.path.exists("Neighborhood_Demographics.csv"):
    df = df.merge(pd.read_csv("Neighborhood_Demographics.csv"), on="location", how="left")

# 4. Filter (Keep built homes, drop land plots)
df = df[~df["area_type"].str.lower().str.contains("plot")]

# 7. Outliers (IQR clipping for price & total_sqft)
print("====== BEFORE OUTLIER CLIPPING =======")
print(df[["price", "total_sqft"]].describe().loc[["min", "max", "mean"]])

for col in ["price", "total_sqft"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    df[col] = df[col].clip(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))

print("\n====== AFTER OUTLIER CLIPPING =======")
print(df[["price", "total_sqft"]].describe().loc[["min", "max", "mean"]])

# 6. Aggregate & Feature Engineering
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

# 5. Handle Categorical Variables (Grouping & Encoding)
print("==== BEFORE CATEGORICAL GROUPING =====")
print(f"Total Unique Locations: {df['location'].nunique()}")

# Group rare locations into "other" to prevent column explosion
counts = df["location"].value_counts()
df.loc[df["location"].isin(counts[counts <= 10].index), "location"] = "other"

print("\n==== AFTER CATEGORICAL GROUPING =====")
print(f"Unique Locations (rare ones grouped to 'other'): {df['location'].nunique()}")

# Drop intermediate/redundant columns and one-hot encode
df_encoded = pd.get_dummies(
    df.drop(columns=["size", "price_per_sqft"]), 
    columns=["area_type", "location"], 
    drop_first=True
)

print("======== FINAL CLEANED DATASET =======")
print(f"Final Shape: {df_encoded.shape}")
print("\nFinal Data (first 3 rows):")
print(df_encoded.head(3))