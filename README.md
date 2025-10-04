# Exoplanet System Detective

ML-powered exoplanet detection with 3D visualization, physics validation, and explainable AI.

**91% accuracy | 97% AUC | 5 ML models | Interactive 3D | Habitable zone calculations**

## Quick Start

### 1. Start the Backend (Terminal 1)
```bash
./start_server.sh
```

### 2. Start the Frontend (Terminal 2)
```bash
uv run serve_frontend.py
```

### 3. Open Browser
Navigate to **http://localhost:3000**

## What You Can Do

**🌟 Interactive 3D Exploration**
- Adjust orbital period, transit depth, stellar temperature in the left panel
- Click "Update System" to recalculate everything
- Watch the planet orbit in real-time with accurate physics
- See habitable zone boundaries (green = good for life!)
- Rotate/zoom the view with your mouse

**🔬 Scientific Analysis**
- View planet type, mass, radius, and temperature
- Check if planet is in the habitable zone
- See validation warnings (too hot? wrong density?)
- All calculations use real astrophysics (Kepler's laws, Kopparapu 2013 HZ)

**🤖 ML Predictions** (via API)
- 5-model ensemble with 91% accuracy
- SHAP explanations showing why the model thinks it's a planet
- Confidence scores based on model agreement

## System Features

- 7,463 training samples from NASA Exoplanet Archive
- 46 engineered features (transit depth, stellar parameters, etc.)
- Physics validation (density checks, Roche limit, temperature)
- <100ms API response time
- Real-time 3D rendering at 60 FPS

## Requirements

- Python 3.12+
- Modern browser
- `uv sync` to install dependencies

## Credits

NASA Exoplanet Archive | Kopparapu et al. (2013) | SHAP | Three.js

---
*Built for NASA Space Apps Challenge 2025* 🌍✨
