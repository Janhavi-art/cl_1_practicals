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