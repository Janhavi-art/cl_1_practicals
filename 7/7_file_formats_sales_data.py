import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load data
csv = pd.read_csv('sales_data_sample.csv', encoding='cp1252')
ed = pd.read_excel('sales_data_sample.xlsx')
with open('customers.json', 'r') as json_file:
  json_data = json.load(json_file)
json_df = pd.json_normalize(json_data)

# 2. Explore data
print(csv.info())
print(csv.describe())
print(ed.head())
print(ed.info())

# 3. Clean data
csv = csv.dropna(subset=['SALES']).drop_duplicates()
ed = ed.dropna().drop_duplicates()

# 4. Unify format
unified_data = pd.concat([csv, ed], ignore_index=True)

# 5. Transform data (derive new variable using SALES)
unified_data['TAX'] = unified_data['SALES'] * 0.10
# 6. Analyze sales data
total_sales = unified_data['SALES'].sum()
avg_sales = unified_data['SALES'].mean()
print('Total Sales:', total_sales)
print('Average Order Value:', avg_sales)

order_sales = unified_data.groupby('ORDERNUMBER')['SALES'].sum()
print(order_sales.head())

# 7. Visualizations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
unified_data['SALES'].plot(kind='hist', bins=20, edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('Sales')

plt.subplot(1, 2, 2)
order_sales.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Orders by Sales')
plt.xlabel('Order Number')
plt.ylabel('Sales')

plt.tight_layout()
plt.show()

# Data Integration, Preprocessing and Visualization:
# Data Integration is the process of combining data from multiple sources
# and formats such as CSV, Excel and JSON into a unified dataset.
#
# Data Loading: Pandas read_csv() and read_excel() are used to load CSV
# and Excel files, while json.load() reads JSON data. json_normalize()
# converts nested JSON data into a tabular DataFrame.
#
# Data Exploration is performed to understand the dataset structure,
# data types, missing values and statistical properties.
# info() displays column information, describe() gives descriptive
# statistics (mean, standard deviation, minimum, maximum, etc.),
# and head() displays the first five records.
#
# Data Cleaning involves handling missing values and duplicate records.
# dropna() removes rows containing missing values, while
# drop_duplicates() removes repeated records to improve data quality.
#
# Data Integration is performed using pd.concat() to combine CSV
# and Excel datasets into a single DataFrame.
#
# Data Transformation involves modifying existing data or creating
# new features. Here, a TAX column is derived as 10% of SALES.
#
# Data Analysis uses descriptive statistics and aggregation operations.
# sum() calculates total sales, mean() calculates average sales,
# and groupby() groups records based on ORDERNUMBER to calculate
# total sales for each order.
#
# Data Visualization represents data graphically to identify patterns
# and understand distributions.
# Histogram displays the frequency distribution of sales values.
# Bar Chart compares sales values across different order numbers.
#
# Finally, Matplotlib is used to display both visualizations.
# tight_layout() adjusts plot spacing, while show() displays the graphs.