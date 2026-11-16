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

# Bengaluru House Price Data Wrangling and Preprocessing:
# Data Wrangling is the process of cleaning, transforming, integrating
# and organizing raw data into a structured and usable format.
#
# 1. Data Loading and Exploration:
# Pandas read_csv() loads the Bengaluru house dataset.
# shape displays the number of rows and columns.
# isnull().sum() identifies missing values in each column.
# head() displays the first few records for initial inspection.
#
# 2. Column Name Standardization:
# str.lower() converts column names into lowercase.
# str.strip() removes leading and trailing spaces.
# str.replace() replaces spaces with underscores.
# This ensures consistent column naming.
#
# 3. Missing Value Handling and Data Formatting:
# Unnecessary columns such as society and availability are removed.
# pd.to_numeric() converts total_sqft into numerical format.
# errors='coerce' converts invalid values and ranges into NaN.
# dropna() removes rows with missing values in important columns.
# fillna() replaces missing bath and balcony values with their median.
#
# 4. Data Cleaning and Feature Extraction:
# str.strip() and str.lower() standardize location names.
# str.split() separates the size column into individual parts.
# bhk is extracted from the size column and converted into an integer.
# BHK (Bedroom, Hall, Kitchen) represents the number of bedrooms.
#
# 5. Data Integration (Merging):
# os.path.exists() checks whether the demographic dataset exists.
# pd.merge() combines house data with neighborhood demographic data
# using the common location column.
# A left join retains all records from the main house dataset.
#
# 6. Data Filtering:
# Boolean filtering removes land plots and retains other property types.
# The ~ operator represents logical NOT and reverses the filtering condition.
#
# 7. Outlier Detection and Handling:
# The Interquartile Range (IQR) method identifies extreme values
# in numerical columns such as price and total_sqft.
# Q1 represents the 25th percentile and Q3 represents the 75th percentile.
# IQR = Q3 - Q1.
# Lower Bound = Q1 - 1.5 * IQR.
# Upper Bound = Q3 + 1.5 * IQR.
# clip() limits extreme values within the calculated boundaries
# instead of deleting entire records.
# describe() is used to compare minimum, maximum and mean values
# before and after outlier handling.
#
# 8. Feature Engineering and Aggregation:
# Feature Engineering creates new meaningful variables from existing data.
# price_per_sqft is calculated to determine the price per square foot.
# It is calculated by converting price from lakhs to rupees
# and dividing it by the total area in square feet.
#
# 9. Categorical Variable Grouping:
# nunique() counts the number of unique locations.
# value_counts() calculates the frequency of each location.
# Rare locations having 10 or fewer records are grouped into "other".
# This reduces the number of categories and prevents excessive
# feature creation during One-Hot Encoding.
#
# 10. Categorical Encoding:
# pd.get_dummies() performs One-Hot Encoding by converting categorical
# variables into separate binary (0/1) columns.
# area_type and location are encoded into numerical features.
# drop_first=True removes one category from each encoded variable
# to avoid redundant columns and multicollinearity.
#
# 11. Removal of Intermediate Columns:
# size and price_per_sqft are removed from the final dataset.
# size is redundant because BHK has already been extracted.
# price_per_sqft is removed because it is an intermediate feature
# in this particular preprocessing pipeline.
#
# 12. Final Dataset Verification:
# shape displays the final number of rows and columns.
# head() displays the first three records of the cleaned dataset.
#
# Finally, the raw Bengaluru house dataset is cleaned, integrated,
# transformed and encoded into a structured numerical dataset
# suitable for further statistical analysis and machine learning.