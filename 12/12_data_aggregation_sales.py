import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Import the dataset
df = pd.read_csv("12/customer_shopping_data.csv")

# Task 2: Explore the dataset
print("--- Dataset Info ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())

# Task 3: Identify variables 
# (We will use 'shopping_mall' for region, 'price' for sales amount, and 'category' for product)

# Task 4: Group by region and calculate total sales
region_sales = df.groupby('shopping_mall')['price'].sum()
print("\n--- Total Sales by Region ---")
print(region_sales)

# Task 6: Identify the top-performing region
top_region = region_sales.idxmax()
top_sales = region_sales.max()
print(f"\nTop-Performing Region: {top_region} (Sales: {top_sales})")

# Task 5: Visualize sales distribution by region (Pie chart)
region_sales.plot(kind='pie', title="Sales Distribution by Region", autopct='%1.1f%%')
plt.ylabel("")  # Removes the messy side label from the pie chart
plt.show()

# Task 7: Group by region and category (using pivot_table for a clean setup)
region_category_sales = df.pivot_table(index='shopping_mall', columns='category', values='price', aggfunc='sum')
print("\n--- Sales by Region and Category ---")
print(region_category_sales)

# Task 8: Grouped bar plot to compare regions and categories
region_category_sales.plot(kind='bar', figsize=(10, 6), title="Sales by Region and Category")
plt.ylabel("Total Sales Amount")
plt.xlabel("Region (Shopping Mall)")
plt.show()