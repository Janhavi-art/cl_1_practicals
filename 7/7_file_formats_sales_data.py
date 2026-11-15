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
