# Smart Solar Energy Monitoring & Prediction System
## Powered by Temporal Fusion Transformer (TFT)

---

## Architecture

```
Frontend (HTML/JS)  ──REST──▶  FastAPI Backend  ──▶  TFT Model (PyTorch)
                                      │
                                      ├──▶  Open-Meteo Weather API (free)
                                      └──▶  SQLite Database
```

## TFT Model Components
- **Variable Selection Network (VSN)** — learns which inputs matter most
- **Gated Residual Networks (GRN)** — stable nonlinear feature processing
- **LSTM Encoder (2 layers)** — captures 24-hour temporal dependencies
- **Interpretable Multi-Head Self-Attention (4 heads)** — long-range patterns
- **Quantile Regression Head** — outputs p10 / p50 / p90 (uncertainty bounds)

Input features: `temperature`, `cloud_cover`, `irradiance`, `hour`, `month`,
`solar_angle`, `daylight_factor`, `efficiency_factor`

---

## Setup

### Option A — Run Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `frontend/index.html` in your browser.
Set the API URL to `http://localhost:8000` and click **Connect**.

### Option B — Deploy on Replit (public URL, anyone can access)

1. Create a new Replit project (Python template)
2. Upload the `backend/` folder contents + `.replit`
3. Click **Run** — Replit will install deps and start the server
4. Your backend URL will be `https://<your-repl>.replit.dev`
5. Host `frontend/index.html` on GitHub Pages / Netlify / any static host
6. Enter your Replit URL in the API URL field of the dashboard

### Option C — Deploy on Railway / Render (free tier)

```bash
# Procfile (auto-detected by Railway/Render)
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/metrics` | Dashboard stats + TFT model info |
| GET | `/weather?lat=&lon=` | Weather + TFT hourly forecast |
| POST | `/predict` | TFT prediction with p10/p50/p90 |
| POST | `/readings` | Store a sensor reading |
| GET | `/readings?limit=100` | Fetch stored readings |
| GET | `/simulate` | Generate a simulated 24-hour day |
| GET | `/docs` | Interactive Swagger UI |

### POST /predict — Request Body
```json
{
  "temperature": 32,
  "cloud_cover": 20,
  "irradiance": 750,
  "hour_of_day": 12,
  "month": 6
}
```

### POST /predict — Response
```json
{
  "predicted_kwh": 3.24,
  "lower_bound_kwh": 2.91,
  "upper_bound_kwh": 3.58,
  "confidence": 0.87,
  "model": "Temporal Fusion Transformer"
}
```

---

## Project Structure

```
solar-app/
├── backend/
│   ├── main.py          ← FastAPI app (all routes)
│   ├── ml_model.py      ← TFT model (full implementation)
│   ├── requirements.txt
│   └── start.sh
├── frontend/
│   └── index.html       ← Complete dashboard (no build step needed)
└── .replit
```

---

## Notes

- The TFT trains on synthetic solar data at startup (~60 epochs, ~30-90 sec depending on CPU/GPU).
- Weather data comes from **Open-Meteo** — completely free, no API key needed.
- The SQLite database auto-creates at `backend/solar_data.db`.
- For production, swap SQLite with PostgreSQL (change the `get_db()` function).
