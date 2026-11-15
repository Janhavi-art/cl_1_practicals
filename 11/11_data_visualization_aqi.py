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