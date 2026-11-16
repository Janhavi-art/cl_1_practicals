import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Import the dataset
df = pd.read_csv("customer_shopping_data.csv")

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

# Customer Shopping Data Analysis and Visualization:
# Sales Analysis involves examining sales data to identify revenue patterns,
# top-performing regions and product category performance.
#
# 1. Data Loading: Pandas read_csv() is used to load the customer shopping
# dataset into a DataFrame.
#
# 2. Data Exploration: info() displays column names, data types and
# non-null values, while head() displays the first five records.
# This helps understand the dataset structure and contents.
#
# 3. Variable Identification:
# shopping_mall represents the region, price represents the sales amount,
# and category represents the product category.
# These variables are selected for regional and category-wise sales analysis.
#
# 4. Grouping and Aggregation:
# groupby() groups records according to shopping_mall.
# sum() calculates the total sales for each shopping mall.
# This is an example of data aggregation used to summarize sales information.
#
# 5. Top-Performing Region Identification:
# idxmax() returns the shopping mall with the highest total sales.
# max() returns the maximum sales value.
# This helps identify the region generating the highest revenue.
#
# 6. Sales Distribution Visualization (Pie Chart):
# A pie chart represents the contribution of each shopping mall
# to the overall sales.
# Each slice represents a region's proportion of total sales.
# autopct='%1.1f%%' displays percentage values up to one decimal place.
# plt.ylabel("") removes the unnecessary y-axis label.
#
# 7. Multidimensional Grouping (Pivot Table):
# pivot_table() summarizes data based on multiple categorical variables.
# Here, shopping_mall is used as the index, category as the columns,
# and price as the values.
# aggfunc='sum' calculates total sales for each region-category combination.
# This helps compare product category performance across regions.
#
# 8. Grouped Bar Chart:
# A grouped bar chart compares sales across different shopping malls
# and product categories.
# Each group represents a shopping mall, while individual bars
# represent different product categories.
# It helps identify which categories generate higher sales
# in different regions.
#
# 9. Data Visualization and Customization:
# Matplotlib is used to create and customize pie charts and bar charts.
# Titles, axis labels, figure size and percentage annotations
# improve the readability and interpretation of visualizations.
#
# Finally, sales data is analyzed using grouping, aggregation,
# pivot tables and visualization techniques to identify
# top-performing regions and compare category-wise sales.