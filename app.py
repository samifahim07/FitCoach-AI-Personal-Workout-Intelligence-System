
"""
FitCoach AI - Personal Workout Intelligence System

1. Calories Prediction - XGBoost Regressor
2. Fitness Risk Prediction - CatBoost Classifier
3. Backup Prediction - Formula and Rule Based

Flask application for Render deployment.
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings("ignore")


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

# Supports app.py in either root or Code folder
ROOT = BASE.parent if BASE.name.lower() == "code" else BASE

TEMPLATE_DIR = ROOT / "templates"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)

MODEL_DIRS = [
    ROOT,
    ROOT / "Code",
    ROOT / "Data",
    BASE
]

LOAD_ERR = {}


# ==========================================
# 2. LOAD MACHINE LEARNING MODELS
# ==========================================

def load(*names):
    for folder in MODEL_DIRS:
        for name in names:
            path = folder / name

            if not path.is_file():
                continue

            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)

                print(f"[SUCCESS] Loaded: {path}")
                return model

            except Exception as e:
                LOAD_ERR[str(path)] = str(e)
                print(f"[WARNING] {path}: {e}")

    print(f"[WARNING] Model not found: {names}")
    return None


calorie_model = load(
    "xgboost_model.pkl",
    "model_xgb.pkl"
)

risk_model = load("catboost_model.pkl")

scaler = load("scaler.pkl")


_est = getattr(risk_model, "best_estimator_", risk_model)

RISK_ML = bool(
    risk_model is not None
    and scaler is not None
    and hasattr(_est, "predict_proba")
)

print("Calories Model:",
      "READY" if calorie_model is not None else "MISSING")

print("Risk Model:",
      "READY" if RISK_ML else "RULE-BASED FALLBACK")


# ==========================================
# 3. MODEL FEATURE COLUMNS
# ==========================================

NUM = [
    "Age",
    "Weight (kg)",
    "Height (m)",
    "Max_BPM",
    "Avg_BPM",
    "Resting_BPM",
    "Session_Duration (hours)",
    "Fat_Percentage",
    "Water_Intake (liters)",
    "Workout_Frequency (days/week)",
    "Experience_Level",
    "BMI",
    "Heart_Rate_Reserve",
    "HR_Intensity",
    "Workout_Load",
    "Weekly_Workout_Hours"
]

CALORIE_COLS = NUM + [
    "Gender_Female",
    "Gender_Male",
    "Workout_Type_Cardio",
    "Workout_Type_HIIT",
    "Workout_Type_Strength",
    "Workout_Type_Yoga",
    "Workout_Intensity_High",
    "Workout_Intensity_Low",
    "Workout_Intensity_Moderate"
]

RISK_COLS = NUM + [
    "Gender_Male",
    "Workout_Type_HIIT",
    "Workout_Type_Strength",
    "Workout_Type_Yoga",
    "BMI_Category_Obese",
    "BMI_Category_Overweight",
    "BMI_Category_Underweight",
    "Workout_Intensity_Low",
    "Workout_Intensity_Moderate"
]

LEVELS = {
    0: "Low",
    1: "Moderate",
    2: "High"
}

RANGES = {
    "age": (18, 59),
    "weight": (40, 130),
    "height": (1.5, 2.0),
    "max_bpm": (160, 199),
    "avg_bpm": (120, 169),
    "resting_bpm": (50, 74),
    "duration": (0.5, 2.0),
    "fat": (10, 35),
    "water": (1.5, 3.7),
    "frequency": (2, 5)
}


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

def build_features(d):

    r = {
        "Age": d["age"],
        "Weight (kg)": d["weight"],
        "Height (m)": d["height"],
        "Max_BPM": d["max_bpm"],
        "Avg_BPM": d["avg_bpm"],
        "Resting_BPM": d["resting_bpm"],
        "Session_Duration (hours)": d["duration"],
        "Fat_Percentage": d["fat"],
        "Water_Intake (liters)": d["water"],
        "Workout_Frequency (days/week)": d["frequency"],
        "Experience_Level": d["experience"]
    }

    r["BMI"] = (
        r["Weight (kg)"] / r["Height (m)"] ** 2
    )

    r["Heart_Rate_Reserve"] = (
        r["Max_BPM"] - r["Resting_BPM"]
    )

    r["HR_Intensity"] = (
        r["Avg_BPM"] / r["Max_BPM"]
    )

    r["Workout_Load"] = (
        r["Session_Duration (hours)"]
        * r["Avg_BPM"]
    )

    r["Weekly_Workout_Hours"] = (
        r["Workout_Frequency (days/week)"]
        * r["Session_Duration (hours)"]
    )

    bmi = r["BMI"]

    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal"
    elif bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    hi = r["HR_Intensity"]

    if hi < 0.70:
        intensity = "Low"
    elif hi < 0.85:
        intensity = "Moderate"
    else:
        intensity = "High"

    for g in ("Female", "Male"):
        r[f"Gender_{g}"] = int(
            d["gender"] == g
        )

    for w in ("Cardio", "HIIT", "Strength", "Yoga"):
        r[f"Workout_Type_{w}"] = int(
            d["workout_type"] == w
        )

    for c in ("Obese", "Overweight", "Underweight"):
        r[f"BMI_Category_{c}"] = int(
            bmi_cat == c
        )

    for i in ("High", "Low", "Moderate"):
        r[f"Workout_Intensity_{i}"] = int(
            intensity == i
        )

    return r, bmi_cat, intensity


# ==========================================
# 5. BACKUP CALORIE PREDICTION
# ==========================================

def formula_calories(d):

    hr = d["avg_bpm"]
    w = d["weight"]
    a = d["age"]

    if d["gender"] == "Male":

        per_min = (
            -55.0969
            + 0.6309 * hr
            + 0.1988 * w
            + 0.2017 * a
        ) / 4.184

    else:

        per_min = (
            -20.4022
            + 0.4472 * hr
            - 0.1263 * w
            + 0.074 * a
        ) / 4.184

    return max(per_min, 0) * d["duration"] * 60


# ==========================================
# 6. FITNESS RISK FACTORS
# ==========================================

def risk_factors(r):

    checks = [
        (
            r["BMI"] >= 30,
            "Body mass index is in the obese range"
        ),
        (
            r["Fat_Percentage"] >= 30,
            "Body fat is 30% or higher"
        ),
        (
            r["HR_Intensity"] >= 0.85,
            "Average heart rate is very close to your maximum"
        ),
        (
            r["Resting_BPM"] >= 70,
            "Resting heart rate is 70 or higher"
        ),
        (
            r["Workout_Load"] >= 200,
            "Session workload is heavy (long and intense)"
        )
    ]

    return [
        msg for hit, msg in checks if hit
    ]


# ==========================================
# 7. FITNESS RISK PREDICTION
# ==========================================

def predict_risk(r):

    factors = risk_factors(r)

    if RISK_ML:

        try:
            X = pd.DataFrame([r])[RISK_COLS].astype(float)

            X_scaled = scaler.transform(X.values)

            prediction = risk_model.predict(X_scaled)

            predicted_class = int(
                np.ravel(prediction)[0]
            )

            level = LEVELS[predicted_class]

            return level, "ai", factors

        except Exception as e:
            print("[WARNING] Risk model prediction failed:", e)

    score = len(factors)

    if score <= 1:
        level = "Low"
    elif score == 2:
        level = "Moderate"
    else:
        level = "High"

    return level, "rules", factors


# ==========================================
# 8. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 9. MODEL HEALTH CHECK
# ==========================================

@app.route("/health")
def health():

    return jsonify(
        status="running",
        calorie_model_loaded=calorie_model is not None,
        risk_model_loaded=RISK_ML,
        templates_found=(TEMPLATE_DIR / "index.html").is_file()
    )


# ==========================================
# 10. PREDICTION API
# ==========================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        raw = request.get_json(silent=True) or {}

        fields = [
            "age",
            "weight",
            "height",
            "max_bpm",
            "avg_bpm",
            "resting_bpm",
            "duration",
            "fat",
            "water",
            "frequency",
            "experience"
        ]

        d = {
            k: float(raw[k])
            for k in fields
        }

        d["gender"] = raw["gender"]
        d["workout_type"] = raw["workout_type"]

        if not all(np.isfinite(d[k]) for k in fields):
            raise ValueError("Please enter finite numeric values.")

        if (
            d["height"] <= 0
            or d["max_bpm"] <= 0
            or d["duration"] <= 0
        ):
            raise ValueError(
                "Height, max heart rate and duration must be above zero."
            )

        if d["avg_bpm"] > d["max_bpm"]:
            raise ValueError(
                "Average heart rate cannot be higher than your maximum."
            )

        r, bmi_cat, intensity = build_features(d)

        calories = None
        source = "ai"

        # XGBoost calories prediction
        if calorie_model is not None:

            try:

                X = pd.DataFrame([r])[CALORIE_COLS].astype(float)

                calories = float(
                    np.ravel(calorie_model.predict(X))[0]
                )

                if not np.isfinite(calories):
                    raise ValueError("Invalid model output")

            except Exception as e:

                print("[WARNING] Calorie prediction failed:", e)

        # Fallback if model fails
        if calories is None:

            calories = formula_calories(d)
            source = "formula"

        calories = max(calories, 0)

        # Fitness risk prediction
        level, engine, factors = predict_risk(r)

        rough = any(
            not (lo <= d[k] <= hi)
            for k, (lo, hi) in RANGES.items()
        )

        return jsonify(
            ok=True,
            calories=round(calories),
            calorie_source=source,
            calories_per_hour=round(
                calories / d["duration"]
            ),
            risk=level,
            risk_engine=engine,
            factors=factors,
            bmi=round(r["BMI"], 1),
            bmi_category=bmi_cat,
            intensity=intensity,
            weekly_hours=round(
                r["Weekly_Workout_Hours"], 1
            ),
            rough=rough
        )

    except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:

        print("[INPUT ERROR]", e)

        return jsonify(
            ok=False,
            error=str(e)
        ), 400

    except Exception as e:

        print("[PREDICTION ERROR]", e)

        return jsonify(
            ok=False,
            error="Prediction failed. Please check the server logs."
        ), 500


# ==========================================
# 11. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
