#!/bin/bash

# Start the Exoplanet System Detective API server

echo "=============================="
echo "Exoplanet System Detective API"
echo "=============================="
echo ""
echo "Starting server on http://localhost:8000"
echo ""
echo "Available endpoints:"
echo "  GET  /                       - API information"
echo "  GET  /api/health             - Health check"
echo "  GET  /api/models/performance - Model metrics"
echo "  GET  /api/targets            - Demo targets"
echo "  GET  /api/features           - Feature list"
echo "  POST /api/analyze            - Predict from features"
echo "  POST /api/predict-system     - Complete system prediction"
echo ""
echo "Press Ctrl+C to stop"
echo "=============================="
echo ""

cd backend && uv run uvicorn api:app --reload --port 8000
