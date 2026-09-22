import requests
from typing import Dict, Any, Tuple, Optional

from datetime import datetime, timedelta

def geocode(location: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return result["latitude"], result["longitude"], None
        return None, None, f"Could not resolve location: '{location}'"
    except Exception as e:
        return None, None, f"Geocoding API error: {str(e)}"

def fetch_weather(latitude: float, longitude: float, time_reference: Optional[str] = "today") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # Request hourly data for 2 days to cover both "today" and "tomorrow".
    # Using timezone=auto ensures the hours align with the local timezone.
    hourly_params = "temperature_2m,wind_speed_10m,precipitation,precipitation_probability,uv_index,cloud_cover"
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly={hourly_params}&timezone=auto&forecast_days=2"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" not in data or "time" not in data["hourly"]:
            return None, "Weather data structure is invalid: missing 'hourly' or 'time' key."
            
        hourly_data = data["hourly"]
        times = hourly_data["time"]
        
        if not times:
            return None, "No time data returned by weather API."
            
        # Deterministic representation choice:
        # We use the forecast for 12:00 PM (noon) local time of the requested day.
        # Parse the first timestamp to determine "today" in local time
        first_time_str = times[0]
        today_date = datetime.fromisoformat(first_time_str).date()
        
        target_date = today_date
        
        if time_reference:
            time_ref_lower = time_reference.lower()
            if "tomorrow" in time_ref_lower:
                target_date = today_date + timedelta(days=1)
                
        # Format the target datetime string: "YYYY-MM-DDT12:00"
        target_time_str = f"{target_date.isoformat()}T12:00"
        
        try:
            target_index = times.index(target_time_str)
        except ValueError:
            return None, f"Forecast for {target_time_str} is unavailable in the returned data."
            
        weather_data = {
            "temperature_2m": hourly_data["temperature_2m"][target_index],
            "wind_speed_10m": hourly_data["wind_speed_10m"][target_index],
            "precipitation": hourly_data["precipitation"][target_index],
            "precipitation_probability": hourly_data["precipitation_probability"][target_index],
            "uv_index": hourly_data["uv_index"][target_index],
            "cloud_cover": hourly_data["cloud_cover"][target_index],
        }
        
        return weather_data, None
        
    except Exception as e:
        return None, f"Weather API error: {str(e)}"

def get_weather_for_location(location: str, time_reference: Optional[str] = "today") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    lat, lon, err = geocode(location)
    if err:
        return None, err
    if lat is None or lon is None:
        return None, f"Could not resolve location: '{location}'"
    return fetch_weather(lat, lon, time_reference)
