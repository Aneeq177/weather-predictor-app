# live_data_fetcher.py

import requests
import pandas as pd
from geopy.geocoders import Nominatim

def get_live_weather(city_name: str):
    """
    Fetches live weather data for a given city and formats it for the model.
    """
    try:
        # 1. Get coordinates for the city name
        geolocator = Nominatim(user_agent="weather_app")
        location = geolocator.geocode(city_name)
        if not location:
            print(f"Could not find coordinates for city: {city_name}")
            return None, "City not found."

        lat, lon = location.latitude, location.longitude
        print(f"Found coordinates for {city_name}: Lat={lat}, Lon={lon}")

        # 2. Fetch live weather from Open-Meteo API
        # We ask for the variables our original model was trained on.
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # 3. Format the data to match the model's expected input
        # NOTE: This part is tricky as a live API won't have "Severity" or "AirportCode".
        # We will use default/placeholder values for the features our model expects
        # but the live API doesn't provide.
        live_weather_data = {
            'Severity': 'UNK',  # Default placeholder
            'TimeZone': data['timezone'],
            'AirportCode': 'KNYC', # Default placeholder
            'LocationLat': lat,
            'LocationLng': lon
        }
        
        # We create a single-row DataFrame, just like the model expects
        df = pd.DataFrame([live_weather_data])
        return df, None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, str(e)