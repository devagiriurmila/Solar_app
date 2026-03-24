from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import httpx
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
from ml_model import SolarMLModel
from ml_model import TempForecastModel

app = FastAPI(title="Smart Solar Energy API", version="1.0.0")

# ── CORS: allow any origin so your frontend can call from anywhere ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database setup ──────────────────────────────────────────────────
DB_PATH = "solar_data.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS energy_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            power_kw REAL,
            energy_kwh REAL,
            temperature REAL,
            cloud_cover REAL,
            irradiance REAL,
            predicted_kwh REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── ML Model (singleton) ────────────────────────────────────────────
solar_model = SolarMLModel()
temp_model  = TempForecastModel()


# ── Schemas ─────────────────────────────────────────────────────────
class ReadingInput(BaseModel):
    power_kw: float
    energy_kwh: float
    temperature: Optional[float] = None
    cloud_cover: Optional[float] = None
    irradiance: Optional[float] = None

class PredictInput(BaseModel):
    temperature: float
    cloud_cover: float       # 0–100 %
    irradiance: float        # W/m²
    hour_of_day: int         # 0–23
    month: int               # 1–12

# ── Routes ──────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Smart Solar Energy API", "status": "running", "docs": "/docs"}


@app.get("/weather")
async def get_weather(lat: float = 13.0827, lon: float = 80.2707):
    """
    Fetch current weather + hourly forecast from Open-Meteo (free, no key needed).
    Default coordinates: Chennai, India.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,cloud_cover,direct_radiation"
        "&current=temperature_2m,weather_code,wind_speed_10m"
        "&forecast_days=1"
        "&timezone=auto"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        raw = data.get("current", {})
        current = {"temperature": raw.get("temperature_2m"), "windspeed": raw.get("wind_speed_10m"), "weathercode": raw.get("weather_code", 0)}
        hourly  = data.get("hourly", {})

        # Build 24-hour forecast list
        forecast = []
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        clouds = hourly.get("cloud_cover", [])
        rads   = hourly.get("direct_radiation", [])

        for i, t in enumerate(times):
            hour = int(t[11:13])
            temp  = temps[i]  if i < len(temps)  else 25.0
            cloud = clouds[i] if i < len(clouds) else 20.0
            rad   = rads[i]   if i < len(rads)   else 0.0

            predicted = solar_model.predict(temp, cloud, rad, hour, datetime.now().month)
            forecast.append({
                "time": t,
                "hour": hour,
                "temperature": temp,
                "cloud_cover": cloud,
                "irradiance": rad,
                "predicted_kwh": round(predicted, 3),
            })

        return {
            "location": {"lat": lat, "lon": lon},
            "current_weather": current,
            "forecast": forecast,
            "daily_predicted_kwh": round(sum(f["predicted_kwh"] for f in forecast), 2),
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {str(e)}")


@app.post("/predict")
def predict_energy(data: PredictInput):
    """Run the TFT model to predict solar energy output with uncertainty bounds."""
    bounds = solar_model.predict_with_bounds(
        data.temperature, data.cloud_cover,
        data.irradiance, data.hour_of_day, data.month
    )
    confidence = solar_model.get_confidence(
        data.temperature, data.cloud_cover,
        data.irradiance, data.hour_of_day, data.month
    )
    return {
        "predicted_kwh": bounds["p50"],        # median = point estimate
        "lower_bound_kwh": bounds["p10"],      # 10th percentile
        "upper_bound_kwh": bounds["p90"],      # 90th percentile
        "confidence": round(confidence, 2),
        "model": "Temporal Fusion Transformer",
        "inputs": data.dict(),
    }


@app.post("/readings")
def add_reading(data: ReadingInput):
    """Store a new energy reading from your solar panels."""
    now = datetime.utcnow().isoformat()
    predicted = None
    if data.temperature and data.cloud_cover and data.irradiance:
        predicted = solar_model.predict(
            data.temperature, data.cloud_cover,
            data.irradiance, datetime.utcnow().hour,
            datetime.utcnow().month
        )

    conn = get_db()
    conn.execute(
        """INSERT INTO energy_readings
           (timestamp, power_kw, energy_kwh, temperature, cloud_cover, irradiance, predicted_kwh)
           VALUES (?,?,?,?,?,?,?)""",
        (now, data.power_kw, data.energy_kwh, data.temperature,
         data.cloud_cover, data.irradiance, predicted)
    )
    conn.commit()
    conn.close()
    return {"status": "saved", "timestamp": now, "predicted_kwh": predicted}


@app.get("/readings")
def get_readings(limit: int = 100):
    """Retrieve the latest energy readings."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM energy_readings ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return {"readings": [dict(r) for r in rows]}


@app.get("/metrics")
def get_metrics():
    """Return summary stats: today's total, average power, model accuracy."""
    conn = get_db()
    today = datetime.utcnow().date().isoformat()

    today_total = conn.execute(
        "SELECT COALESCE(SUM(energy_kwh),0) FROM energy_readings WHERE timestamp LIKE ?",
        (f"{today}%",)
    ).fetchone()[0]

    avg_power = conn.execute(
        "SELECT COALESCE(AVG(power_kw),0) FROM energy_readings"
    ).fetchone()[0]

    total_readings = conn.execute(
        "SELECT COUNT(*) FROM energy_readings"
    ).fetchone()[0]

    conn.close()

    model_info = solar_model.get_model_info()

    return {
        "today_energy_kwh": round(today_total, 3),
        "average_power_kw": round(avg_power, 3),
        "total_readings": total_readings,
        "model": model_info,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.api_route("/simulate", methods=["GET", "POST"])
def simulate_day():
    """Generate a simulated full-day reading and store it (for demo/testing)."""
    now = datetime.utcnow()
    inserted = []
    for h in range(24):
        temp = 28 + 5 * np.sin((h - 6) * np.pi / 12)
        cloud = max(0, min(100, 20 + 30 * np.random.random()))
        irrad = max(0, 900 * np.sin((h - 6) * np.pi / 12) * (1 - cloud / 150))
        power = max(0, irrad / 1000 * 5 * (1 - cloud / 200))

        predicted = solar_model.predict(temp, cloud, irrad, h, now.month)
        ts = (now - timedelta(hours=(23 - h))).isoformat()

        conn = get_db()
        conn.execute(
            """INSERT INTO energy_readings
               (timestamp, power_kw, energy_kwh, temperature, cloud_cover, irradiance, predicted_kwh)
               VALUES (?,?,?,?,?,?,?)""",
            (ts, round(power, 3), round(power, 3),
             round(float(temp), 1), round(float(cloud), 1),
             round(float(irrad), 1), round(predicted, 3))
        )
        conn.commit()
        conn.close()
        inserted.append({"hour": h, "power_kw": round(power, 3)})

    return {"status": "simulated", "readings": inserted}

# ── 7-Day Forecast from Historical CSV Data ─────────────────────────
import csv, re

def parse_csv_data():
    """Parse data.csv into daily aggregates: avg temp, avg irradiance, avg current."""
    path = "data.csv"
    if not os.path.exists(path):
        return []

    def extract_num(val):
        if val is None: return None
        m = re.search(r"[-+]?\d*\.?\d+", str(val))
        return float(m.group()) if m else None

    daily = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("created_at", "")
            date = ts[:10]
            if not date or date == "created_at":
                continue
            temp    = extract_num(row.get("temp"))
            humidity= extract_num(row.get("humidity"))
            ldr1    = extract_num(row.get("ldr1"))
            ldr2    = extract_num(row.get("ldr2"))
            current = extract_num(row.get("current"))
            hour    = int(ts[11:13]) if len(ts) > 12 else 12

            if date not in daily:
                daily[date] = {"temps":[], "humidities":[], "irradiances":[], "currents":[], "hours":[]}
            if temp is not None:    daily[date]["temps"].append(temp)
            if humidity is not None:daily[date]["humidities"].append(humidity)
            if ldr1 and ldr2:       daily[date]["irradiances"].append((ldr1 + ldr2) / 2 * 100)
            if current is not None: daily[date]["currents"].append(current)
            daily[date]["hours"].append(hour)

    result = []
    for date, vals in sorted(daily.items()):
        result.append({
            "date":        date,
            "avg_temp":    round(np.mean(vals["temps"]),    2) if vals["temps"]    else 30.0,
            "avg_humidity":round(np.mean(vals["humidities"]),1) if vals["humidities"] else 60.0,
            "avg_irradiance": round(np.mean(vals["irradiances"]), 1) if vals["irradiances"] else 400.0,
            "avg_current": round(np.mean(vals["currents"]), 3) if vals["currents"] else 1.0,
            "n_readings":  len(vals["currents"]),
        })
    return result


@app.get("/forecast/7day")
def forecast_7day():
    """
    Predict next 7 days of solar energy output using patterns from the historical CSV dataset.
    Uses day-of-week and monthly seasonality derived from your real sensor data.
    """
    records = parse_csv_data()
    if not records:
        raise HTTPException(status_code=404, detail="data.csv not found or empty")

    today = datetime.utcnow().date()
    arr   = np.array([[r["avg_temp"], r["avg_humidity"], r["avg_irradiance"], r["avg_current"]]
                      for r in records], dtype=np.float32)

    # Compute per-month averages from historical data
    month_stats = {}
    for r in records:
        m = int(r["date"][5:7])
        if m not in month_stats:
            month_stats[m] = []
        month_stats[m].append(r["avg_irradiance"])

    month_avg_irr = {m: float(np.mean(v)) for m, v in month_stats.items()}
    global_avg_irr = float(np.mean(arr[:, 2]))

    # Day-of-week bias from historical data
    dow_stats = {i: [] for i in range(7)}
    for r in records:
        try:
            from datetime import date as date_cls
            d = date_cls.fromisoformat(r["date"])
            dow_stats[d.weekday()].append(r["avg_irradiance"])
        except: pass
    dow_avg = {k: float(np.mean(v)) if v else global_avg_irr for k, v in dow_stats.items()}

    # Recent trend: last 7 days avg vs overall avg
    recent = arr[-7:, 2] if len(arr) >= 7 else arr[:, 2]
    trend_factor = float(np.mean(recent)) / global_avg_irr if global_avg_irr > 0 else 1.0
    trend_factor = float(np.clip(trend_factor, 0.5, 1.5))

    forecast = []
    for d in range(1, 8):
        future_date = today + timedelta(days=d)
        month       = future_date.month
        dow         = future_date.weekday()

        # Blend: monthly seasonal pattern + day-of-week + recent trend
        monthly_irr = month_avg_irr.get(month, global_avg_irr)
        dow_irr     = dow_avg.get(dow, global_avg_irr)
        blended_irr = (monthly_irr * 0.5 + dow_irr * 0.3 + global_avg_irr * 0.2) * trend_factor

        # Recent avg temp as base
        avg_temp    = float(np.mean(arr[-14:, 0])) if len(arr) >= 14 else float(np.mean(arr[:, 0]))
        avg_humidity= float(np.mean(arr[-14:, 1])) if len(arr) >= 14 else float(np.mean(arr[:, 1]))
        cloud_est   = float(np.clip(avg_humidity * 0.6, 10, 90))   # humidity → cloud proxy

        # Get TFT predictions for peak hour (12) and compute daily total
        daily_kwh = 0.0
        hourly = []
        for h in range(24):
            solar_frac = max(0.0, float(np.sin((h - 6) * np.pi / 12)))
            irr_h = blended_irr * solar_frac
            kwh_h = solar_model.predict(avg_temp, cloud_est, irr_h, h, month)
            daily_kwh += kwh_h
            if 6 <= h <= 20:
                hourly.append({"hour": h, "predicted_kwh": round(kwh_h, 3), "irradiance": round(irr_h, 1)})

        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        forecast.append({
            "date":            future_date.isoformat(),
            "day":             day_names[dow],
            "predicted_kwh":   round(daily_kwh, 2),
            "avg_temp":        round(avg_temp, 1),
            "cloud_estimate":  round(cloud_est, 1),
            "avg_irradiance":  round(blended_irr, 1),
            "hourly":          hourly,
        })

    return {
        "forecast": forecast,
        "based_on_days": len(records),
        "data_range": {"from": records[0]["date"], "to": records[-1]["date"]},
        "trend_factor": round(trend_factor, 3),
    }


# ── 7-Day Temperature & Humidity Forecast ───────────────────────────
import csv as csv_module, re as re_module

def load_sensor_history():
    """Load recent sensor readings from data.csv for calibration."""
    path = "data.csv"
    if not os.path.exists(path):
        return []
    def extract_num(val):
        m = re_module.search(r"[-+]?\d*\.?\d+", str(val))
        return float(m.group()) if m else None
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            ts   = row.get("created_at", "")
            temp = extract_num(row.get("temp"))
            hum  = extract_num(row.get("humidity"))
            if temp and hum:
                rows.append({"temp": temp, "humidity": hum, "ts": ts})
    return rows


@app.get("/forecast/temperature")
def forecast_temperature(days: int = 7):
    """
    Predict temperature & humidity for the next N days (default 7),
    every 3 hours per day (8 readings/day = 56 total).
    Uses TFT trained on Chennai climate + calibrated with your sensor data.
    """
    sensor_history = load_sensor_history()
    start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    predictions = temp_model.predict_7days(start_date, sensor_history)
    if not predictions:
        raise HTTPException(status_code=500, detail="Model not ready")

    # Group by day
    from collections import defaultdict as _dd
    by_day = _dd(list)
    for p in predictions:
        by_day[p["date"]].append(p)

    daily_summary = []
    for date, readings in sorted(by_day.items()):
        temps = [r["temperature"] for r in readings]
        hums  = [r["humidity"]    for r in readings]
        daily_summary.append({
            "date":      date,
            "day_name":  readings[0]["day_name"],
            "temp_min":  round(min(temps), 1),
            "temp_max":  round(max(temps), 1),
            "temp_avg":  round(sum(temps) / len(temps), 1),
            "hum_avg":   round(sum(hums)  / len(hums),  1),
            "readings":  readings,
        })

    return {
        "forecast":        daily_summary,
        "total_readings":  len(predictions),
        "start_date":      start_date.strftime("%Y-%m-%d"),
        "calibrated_from": len(sensor_history),
        "model":           "TFT Temperature Forecaster (Chennai-calibrated)",
    }