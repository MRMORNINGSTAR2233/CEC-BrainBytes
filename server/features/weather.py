import requests
from typing import Dict, Any

class WeatherScraper:
    def __init__(self):
        """
        Initialize the WeatherScraper with Open-Meteo API endpoint.
        """
        self.base_url = "https://api.open-meteo.com/v1"
        
    def get_location(self) -> Dict[str, float]:
        """
        Get Udupi's location coordinates.
        """
        return {
            'lat': 13.3409,
            'lon': 74.7421,
            'city': 'Udupi',
            'country': 'India'
        }

    def get_current_weather(self) -> Dict[str, Any]:
        """
        Get current weather data for Udupi.
        """
        location = self.get_location()
        endpoint = f"{self.base_url}/forecast"
        params = {
            'latitude': location['lat'],
            'longitude': location['lon'],
            'current': 'temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,wind_speed_10m,weather_code',
            'timezone': 'Asia/Kolkata'
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching weather data: {str(e)}")

def get_weather_description(code: int) -> str:
    """
    Convert WMO weather codes to descriptions.
    """
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_codes.get(code, "Unknown")

def format_weather_data(weather_data: Dict[str, Any]) -> str:
    """
    Format weather data into a readable string.
    """
    current = weather_data['current']
    weather_code = current['weather_code']
    
    return f"""
Current Weather in Udupi:
Temperature: {current['temperature_2m']}°C
Feels like: {current['apparent_temperature']}°C
Humidity: {current['relative_humidity_2m']}%
Pressure: {current['pressure_msl']} hPa
Wind Speed: {current['wind_speed_10m']} km/h
Weather: {get_weather_description(weather_code)}
"""

def main():
    try:
        # Initialize the weather scraper
        scraper = WeatherScraper()
        
        # Get current weather
        print("Fetching current weather for Udupi...")
        current_weather = scraper.get_current_weather()
        print(format_weather_data(current_weather))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
