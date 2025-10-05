# Exoplanet System Detective 🪐

AI-powered exoplanet detection system with interactive 3D visualization. Analyzes transit signals using machine learning and renders complete planetary systems in real-time.

![System Status](https://img.shields.io/badge/accuracy-91%25-brightgreen)
![AUC](https://img.shields.io/badge/AUC-97%25-blue)

## Features

- **ML Ensemble**: 5 models (LightGBM, XGBoost, Random Forest, AdaBoost, Extra Trees) vote with SHAP explainability
- **Physics Engine**: Calculates orbits, habitable zones, and validates physics
- **3D Visualization**: Real-time rendering with accurate stellar colors and orbital mechanics
- **Interactive Demos**: Pre-loaded examples including Hot Jupiters and Mini-Neptunes

## Quick Start

### Prerequisites
- Python 3.12+
- uv (Python package manager)

### Installation

```bash
# Install dependencies
uv sync

# Start backend API (port 8000)
./start_server.sh

# Start frontend (port 3000)
uv run serve_frontend.py
```

Open http://localhost:3000 in your browser.

## Usage

### Try Demo Examples
1. Select from dropdown: Hot Jupiter, Super Earth ,or, Mini-Neptune
2. Click "Update System"
3. View ML predictions, model votes, SHAP features, and 3D visualization

### Custom Analysis
1. Input transit parameters:
   - Orbital period (days)
   - Transit depth (ppm)
   - Transit duration (hours)
   - Stellar temperature, radius, surface gravity
2. Click "Update System"
3. Explore results

## API Endpoints

```bash
# Health check
curl http://localhost:8000/api/health

# Predict system
curl -X POST http://localhost:8000/api/predict-system \
  -H "Content-Type: application/json" \
  -d '{"period_days": 24.0, "transit_depth_ppm": 1200, ...}'

# Get demo targets
curl http://localhost:8000/api/targets

# Model performance
curl http://localhost:8000/api/models/performance
```

## Project Structure

```
space-apps/
├── backend/           # FastAPI server, ML models, physics engine
├── frontend/          # Three.js 3D visualization
├── models/           # Trained models (13 MB total)
├── data/             # Kepler mission data (7,463 targets)
└── visualizations/   # Performance plots
```

## Performance

- **Accuracy**: 91% on test set
- **AUC**: 97%
- **Response Time**: <100ms average
- **Concurrent Handling**: 5 requests with 100% success
- **3D Rendering**: 60 FPS

## Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, LightGBM, XGBoost, SHAP
- **Frontend**: Three.js (WebGL), JavaScript ES6
- **Physics**: Kepler's laws, Kopparapu habitable zones
- **Data**: NASA Kepler mission (2,746 confirmed planets)

## Credits

- NASA Exoplanet Archive (KOI & TOI catalogs)
- Kopparapu et al. (2013) - Habitable zone calculations
- Weiss & Marcy (2014) - Mass-radius relations
- SHAP library - Model explainability
