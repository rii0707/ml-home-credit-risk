"""
app/app.py — Optional FastAPI serving layer.

Loads the trained ensemble weights and a pre-fitted LightGBM model,
then exposes a /predict endpoint.

Usage:
    pip install fastapi uvicorn
    uvicorn app.app:app --reload
"""

# Uncomment below once fastapi / uvicorn are installed.

# import pickle
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
#
# MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
#
# app = FastAPI(title="Home Credit Default Risk API")
#
# with open(MODELS_DIR / "training_results.pkl", "rb") as f:
#     _results = pickle.load(f)
#
# _weights = _results["blend_weights"]
#
#
# class LoanApplication(BaseModel):
#     features: dict  # feature_name → value
#
#
# @app.get("/health")
# def health():
#     return {"status": "ok"}
#
#
# @app.post("/predict")
# def predict(application: LoanApplication):
#     df = pd.DataFrame([application.features])
#     # Replace with loaded, fitted pipelines for production use.
#     return {"default_probability": None, "message": "wire up a fitted model here"}

print("app.py loaded — uncomment FastAPI block and install fastapi/uvicorn to serve.")
