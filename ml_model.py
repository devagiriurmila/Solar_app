# ml_model.py

import numpy as np
from datetime import datetime, timedelta

# ── Solar Energy ML Model ──────────────────────────────────────────────
class SolarMLModel:
    def __init__(self):
        # Here you can load your trained model using joblib/pickle
        # Example: self.model = joblib.load("solar_model.pkl")
        pass

    def predict(self, temperature, cloud_cover, irradiance, hour, month):
        """
        Simple dummy prediction:
        predicted kWh = irradiance factor * temperature factor * (1 - cloud cover factor)
        """
        temp_factor = max(0, temperature / 30)       # normalize temp
        cloud_factor = max(0, 1 - cloud_cover / 100)
        hour_factor = max(0, np.sin((hour - 6) * np.pi / 12))  # solar pattern
        predicted = irradiance / 1000 * temp_factor * cloud_factor * hour_factor
        return round(predicted, 3)

    def predict_with_bounds(self, temperature, cloud_cover, irradiance, hour, month):
        """Return dummy lower, median, upper bounds."""
        p50 = self.predict(temperature, cloud_cover, irradiance, hour, month)
        return {"p10": round(p50*0.9, 3), "p50": p50, "p90": round(p50*1.1, 3)}

    def get_confidence(self, temperature, cloud_cover, irradiance, hour, month):
        """Dummy confidence."""
        return 0.9

    def get_model_info(self):
        return {"name": "Dummy Solar Model", "version": "1.0"}

# ── Temperature & Humidity Forecast Model ─────────────────────────────
class TempForecastModel:
    def __init__(self):
        # Here you can load a trained LSTM model if available
        pass

    def predict_7days(self, start_date, sensor_history):
        """
        Generate dummy temperature & humidity forecasts every 3 hours for 7 days.
        start_date = datetime object for day 1
        sensor_history = list of previous readings (not used here, just for API compatibility)
        """
        results = []
        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        for d in range(7):
            day_date = (start_date + timedelta(days=d)).date().isoformat()
            day_name = day_names[(start_date + timedelta(days=d)).weekday()]
            for hour in range(0, 24, 3):  # every 3 hours
                temp = 28 + 4 * np.sin((hour-6) * np.pi / 12)  # simple sinusoidal temp
                hum  = 60 + 20 * np.cos((hour-6) * np.pi / 12) # simple sinusoidal humidity
                results.append({
                    "date": day_date,
                    "day_name": day_name,
                    "hour": hour,
                    "temperature": round(temp, 1),
                    "humidity": round(hum, 1)
                })
        return results