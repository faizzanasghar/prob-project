import requests
import logging

def fetch_realtime_weather(lat: float, lon: float) -> dict:
    """
    Fetch current weather from Open-Meteo for a given location.
    Returns a dict compatible with our model input.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}"}
        
        data = response.json()
        current = data.get("current", {})
        
        # Map to our feature names
        return {
            "tavg": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "pressure": current.get("surface_pressure"),
            "wind_speed": current.get("wind_speed_10m"),
            "cloud_cover": current.get("cloud_cover"),
            "sunshine_hours": 8.0, # Estimate since it's not in 'current' easily
            "source": "Open-Meteo Live"
        }
    except Exception as e:
        return {"error": str(e)}
