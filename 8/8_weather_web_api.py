import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium 

api_key = "fb365aa6104829b44455572365ff3b4e"
lat, lon = 18.184135, 74.610764

# Tasks 1 & 2: Get data
url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
data = requests.get(url).json()
print(data)

# Task 3: Extract attributes (including precipitation)
weather = pd.DataFrame({
    'Timestamp': [pd.to_datetime(i['dt'], unit='s') for i in data['list']],
    'Temperature': [i['main']['temp'] for i in data['list']],
    'humidity': [i['main']['humidity'] for i in data['list']],
    'wind_speed': [i['wind']['speed'] for i in data['list']],
    'precipitation': [i.get('rain', {}).get('3h', 0) for i in data['list']]
}).set_index('Timestamp')

# Task 4: Clean data
weather.fillna(0, inplace=True)

# Task 5: Data modeling (Min/Max)
print("Max Temp:", weather['Temperature'].max())
print("Min Temp:", weather['Temperature'].min())

# Tasks 6 & 7: Visualize and Aggregate (Daily)
weather['Temperature'].resample('D').mean().plot()
plt.title("Daily Mean Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.show()

weather['precipitation'].resample('D').sum().plot(kind='bar')
plt.title("Daily Total Precipitation")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.show()

# Task 9: Relationships
plt.scatter(weather['Temperature'], weather['wind_speed'])
plt.title("Temperature vs Wind Speed")
plt.xlabel("Temperature (°C)")
plt.ylabel("Wind Speed (m/s)")
plt.show()

sns.heatmap(weather[['Temperature', 'humidity', 'precipitation']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Task 8: Geospatial Visualization
weather_map = folium.Map(location=[lat, lon], zoom_start=10)
folium.Marker([lat, lon], popup=f"Temp: {weather['Temperature'].iloc[0]}°C").add_to(weather_map)
weather_map 

# Weather Data Analysis and Geospatial Visualization:
# Weather Data Analysis involves collecting, processing, analyzing and
# visualizing weather information such as temperature, humidity,
# wind speed and precipitation.
#
# API (Application Programming Interface) allows two applications
# to communicate and exchange data. Here, the OpenWeatherMap API
# is used to retrieve weather forecast data in JSON format.
#
# Latitude and Longitude represent the geographical coordinates
# of the location for which weather information is retrieved.
#
# Data Extraction: The requests.get() method sends an HTTP GET request
# to the weather API, and .json() converts the API response into
# a Python dictionary.
#
# DataFrame Creation: Pandas DataFrame is used to organize extracted
# weather attributes such as temperature, humidity, wind speed
# and precipitation into a tabular structure.
#
# Timestamp Conversion: Unix timestamps are converted into
# human-readable date and time format using pd.to_datetime().
#
# Precipitation represents the amount of rainfall. The get() method
# with default value 0 handles missing rainfall information.
#
# Data Cleaning: fillna(0) replaces missing values with zero
# to handle incomplete weather records.
#
# Data Modeling (Statistical Analysis): Minimum and maximum functions
# are used to identify the lowest and highest recorded temperatures.
#
# Time Series Analysis: Resampling is used to aggregate time-based
# weather data into daily intervals.
# resample('D').mean() calculates the daily average temperature.
# resample('D').sum() calculates the daily total precipitation.
#
# Data Visualization: Matplotlib is used to represent weather data
# graphically and identify patterns and variations over time.
# Line Plot displays daily mean temperature trends.
# Bar Chart displays daily total precipitation.
# Scatter Plot represents the relationship between temperature
# and wind speed, where each point represents an observation.
#
# Correlation Analysis: Correlation measures the strength and direction
# of the linear relationship between numerical variables.
# The correlation coefficient ranges from -1 to +1.
# Seaborn heatmap visualizes the correlation matrix using colours.
# annot=True displays the correlation values inside the heatmap.
#
# Geospatial Visualization: Folium is used to display geographical
# information on an interactive map using latitude and longitude.
# folium.Map() creates the map, while folium.Marker() adds a location
# marker with a popup displaying the temperature.
#
# Finally, the weather data is analyzed using statistical methods,
# time-series visualization, correlation analysis and an interactive
# geographical map.