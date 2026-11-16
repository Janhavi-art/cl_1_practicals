import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Import dataset
df = pd.read_csv("air_quality.csv")

# Task 2: Explore dataset
print(df.info())
print(df.head())

# Task 3: Identify variables and setup time series
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Task 4: Overall AQI trend (Line plot)
df['AQI'].plot(color='purple')
plt.title("Overall AQI Trend Over Time")
plt.ylabel("AQI")
plt.show()

# Task 5: Individual pollutant levels (Line plots)
df[['PM2.5', 'PM10', 'CO']].plot()
plt.title("Pollutant Levels Over Time")
plt.ylabel("Concentration")
plt.show()

# Task 6: Compare across time periods (Bar plot, aggregated by Year)
# 'YE' stands for Year-End frequency
df['AQI'].resample('YE').mean().plot(kind='bar', color='orange')
plt.title("Average Yearly AQI")
plt.ylabel("AQI")
plt.show()

# Task 7: Distribution (Box plot)
df[['PM2.5', 'PM10', 'CO', 'AQI']].plot(kind='box')
plt.title("Distribution of Air Quality Metrics")
plt.ylabel("Value")
plt.show()

# Task 8: Relationship (Scatter plot)
plt.scatter(df['PM2.5'], df['AQI'], color='red', alpha=0.5)
plt.title("AQI vs PM2.5")
plt.xlabel("PM2.5 Level")
plt.ylabel("AQI")
plt.show()

# Task 9: (Covered implicitly by all the custom titles, labels, colors, and transparency above)

# Air Quality Data Analysis and Visualization:
# Air Quality Analysis involves studying air pollution levels using
# parameters such as AQI, PM2.5, PM10 and CO.
#
# 1. Data Loading: Pandas read_csv() is used to load the air_quality
# dataset into a DataFrame.
#
# 2. Data Exploration: info() displays column names, data types and
# non-null values, while head() displays the first five records.
# This helps understand the structure and contents of the dataset.
#
# 3. Time Series Data Preparation: Time Series Analysis deals with
# data collected over time to identify trends and variations.
# pd.to_datetime() converts the Date column into datetime format.
# set_index() sets Date as the DataFrame index for time-based analysis.
#
# 4. Overall AQI Trend (Line Plot):
# A line plot represents changes in AQI over time.
# It helps identify increasing or decreasing air pollution trends.
# AQI (Air Quality Index) indicates the level of air pollution
# and its potential impact on human health.
#
# 5. Individual Pollutant Analysis (Line Plot):
# PM2.5, PM10 and CO pollutant concentrations are plotted over time.
# PM2.5 refers to particulate matter with diameter 2.5 micrometres
# or smaller, while PM10 refers to particles 10 micrometres or smaller.
# CO represents Carbon Monoxide.
# Line plots help compare pollutant variations over time.
#
# 6. Time-Based Aggregation (Bar Plot):
# resample('YE') groups time-series observations into yearly intervals.
# mean() calculates the average AQI for each year.
# A bar plot compares average AQI values across different years.
# This helps identify long-term changes in air quality.
#
# 7. Distribution Analysis (Box Plot):
# A box plot represents the statistical distribution of numerical data.
# It displays the median, first quartile (Q1), third quartile (Q3)
# and potential outliers.
# It helps understand the spread and variability of AQI and pollutant levels.
#
# 8. Relationship Analysis (Scatter Plot):
# A scatter plot represents the relationship between two numerical variables.
# Here, PM2.5 concentration is plotted against AQI.
# Each point represents one observation.
# It helps identify positive, negative or no correlation between variables.
# alpha controls the transparency of plotted points.
#
# 9. Data Visualization and Customization:
# Matplotlib is used to create line plots, bar plots, box plots and scatter plots.
# Titles, axis labels, colours and transparency improve readability
# and make visualizations easier to interpret.
#
# Finally, different visualization techniques are used to study
# air quality trends, pollutant distributions, yearly variations
# and relationships between air pollution parameters.