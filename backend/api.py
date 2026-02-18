from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# scaler fitted only on these 5 in your notebook
scale_cols = ['area','bedrooms','bathrooms','stories','parking']

@app.get("/")
def home():
    return {"message": "House Price API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # 1) yes/no -> 1/0
    binary_cols = [
        'mainroad','guestroom','basement',
        'hotwaterheating','airconditioning','prefarea'
    ]
    for c in binary_cols:
        df[c] = df[c].replace({'yes': 1, 'no': 0})

    # 2) Furnishing -> dummy columns (like your notebook)
    df["furnishing_encoded"] = df["furnishingstatus"].map({
        "unfurnished": 0,
        "semi-furnished": 1,
        "furnished": 2
    })
    df = df.drop("furnishingstatus", axis=1)

    df["furnishing_encoded_1"] = (df["furnishing_encoded"] == 1).astype(int)
    df["furnishing_encoded_2"] = (df["furnishing_encoded"] == 2).astype(int)
    df = df.drop("furnishing_encoded", axis=1)

    # 3) Feature engineering (model expects these)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["area_per_room"] = df["area"] / df["total_rooms"].replace(0, np.nan)
    df["bath_per_room"] = df["bathrooms"] / df["total_rooms"].replace(0, np.nan)
    df["stories_area"] = df["stories"] * df["area"]

    df = df.fillna(0)

    # 4) Ensure expected columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # 5) Reorder exactly like training
    df = df[feature_cols]

    # 6) Scale ONLY the 5 cols scaler knows
    df[scale_cols] = scaler.transform(df[scale_cols])

    # 7) Predict + reverse log
    log_pred = model.predict(df)[0]
    price_pred = float(np.expm1(log_pred))

    return {"predicted_price": price_pred}

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
