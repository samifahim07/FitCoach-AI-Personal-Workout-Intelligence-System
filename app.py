"""FitCoach AI - one Flask app, two models.

1) Calories model  : model_xgb.pkl        (XGBoost regressor)
2) Fitness-risk    : catboost_model.pkl   (CatBoost classifier) + scaler.pkl (RobustScaler)

If the classifier or its scaler is missing/invalid, the app falls back to the
same risk rules used to create the labels in the notebook, so it never crashes.
Run:  python app.py   ->  http://127.0.0.1:5000
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE)  # index.html sits next to app.py


LOAD_ERR = {}


def load(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        LOAD_ERR[name] = f"{type(e).__name__}: {e}"
        print(f"[WARN] could not load {name} -> {LOAD_ERR[name]}")
        return None


calorie_model = load("model_xgb.pkl")
risk_model = load("catboost_model.pkl")
scaler = load("scaler.pkl")

# The classifier is usable only if it really is a classifier AND we have its scaler.
_est = getattr(risk_model, "best_estimator_", risk_model)
RISK_ML = bool(risk_model is not None and scaler is not None and hasattr(_est, "predict_proba"))
if calorie_model is None:
    print("[WARN] Calorie model not loaded - using the backup heart-rate formula.")
    print("       Fix: pip install -U xgboost scikit-learn   (then restart the app)")
print("Calories model:", "ready" if calorie_model is not None else "MISSING")
print("Risk engine   :", "CatBoost classifier" if RISK_ML else "notebook rules (classifier/scaler not found)")

# Column order used when the models were trained (taken from the notebooks)
NUM = ["Age", "Weight (kg)", "Height (m)", "Max_BPM", "Avg_BPM", "Resting_BPM",
       "Session_Duration (hours)", "Fat_Percentage", "Water_Intake (liters)",
       "Workout_Frequency (days/week)", "Experience_Level", "BMI",
       "Heart_Rate_Reserve", "HR_Intensity", "Workout_Load", "Weekly_Workout_Hours"]
CALORIE_COLS = NUM + ["Gender_Female", "Gender_Male", "Workout_Type_Cardio", "Workout_Type_HIIT",
                      "Workout_Type_Strength", "Workout_Type_Yoga", "Workout_Intensity_High",
                      "Workout_Intensity_Low", "Workout_Intensity_Moderate"]
RISK_COLS = NUM + ["Gender_Male", "Workout_Type_HIIT", "Workout_Type_Strength", "Workout_Type_Yoga",
                   "BMI_Category_Obese", "BMI_Category_Overweight", "BMI_Category_Underweight",
                   "Workout_Intensity_Low", "Workout_Intensity_Moderate"]
LEVELS = {0: "Low", 1: "Moderate", 2: "High"}

# Ranges seen in the training data (used only to show a gentle "rough estimate" note)
RANGES = {"age": (18, 59), "weight": (40, 130), "height": (1.5, 2.0), "max_bpm": (160, 199),
          "avg_bpm": (120, 169), "resting_bpm": (50, 74), "duration": (0.5, 2.0),
          "fat": (10, 35), "water": (1.5, 3.7), "frequency": (2, 5)}


def build_features(d):
    """Same feature engineering as the notebooks."""
    r = {
        "Age": d["age"], "Weight (kg)": d["weight"], "Height (m)": d["height"],
        "Max_BPM": d["max_bpm"], "Avg_BPM": d["avg_bpm"], "Resting_BPM": d["resting_bpm"],
        "Session_Duration (hours)": d["duration"], "Fat_Percentage": d["fat"],
        "Water_Intake (liters)": d["water"], "Workout_Frequency (days/week)": d["frequency"],
        "Experience_Level": d["experience"],
    }
    r["BMI"] = r["Weight (kg)"] / r["Height (m)"] ** 2
    r["Heart_Rate_Reserve"] = r["Max_BPM"] - r["Resting_BPM"]
    r["HR_Intensity"] = r["Avg_BPM"] / r["Max_BPM"]
    r["Workout_Load"] = r["Session_Duration (hours)"] * r["Avg_BPM"]
    r["Weekly_Workout_Hours"] = r["Workout_Frequency (days/week)"] * r["Session_Duration (hours)"]

    bmi = r["BMI"]
    bmi_cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    hi = r["HR_Intensity"]
    intensity = "Low" if hi < 0.70 else "Moderate" if hi < 0.85 else "High"

    for g in ("Female", "Male"):
        r[f"Gender_{g}"] = int(d["gender"] == g)
    for w in ("Cardio", "HIIT", "Strength", "Yoga"):
        r[f"Workout_Type_{w}"] = int(d["workout_type"] == w)
    for c in ("Obese", "Overweight", "Underweight"):
        r[f"BMI_Category_{c}"] = int(bmi_cat == c)
    for i in ("High", "Low", "Moderate"):
        r[f"Workout_Intensity_{i}"] = int(intensity == i)
    return r, bmi_cat, intensity


def formula_calories(d):
    """Backup: standard heart-rate based estimate (Keytel et al.) used only if the AI model is unavailable."""
    hr, w, a = d["avg_bpm"], d["weight"], d["age"]
    if d["gender"] == "Male":
        per_min = (-55.0969 + 0.6309 * hr + 0.1988 * w + 0.2017 * a) / 4.184
    else:
        per_min = (-20.4022 + 0.4472 * hr - 0.1263 * w + 0.074 * a) / 4.184
    return max(per_min, 0) * d["duration"] * 60


def risk_factors(r):
    """The 5 checks from the notebook's Risk_Score (plain-language messages)."""
    checks = [
        (r["BMI"] >= 30, "Body mass index is in the obese range"),
        (r["Fat_Percentage"] >= 30, "Body fat is 30% or higher"),
        (r["HR_Intensity"] >= 0.85, "Average heart rate is very close to your maximum"),
        (r["Resting_BPM"] >= 70, "Resting heart rate is 70 or higher"),
        (r["Workout_Load"] >= 200, "Session workload is heavy (long and intense)"),
    ]
    return [msg for hit, msg in checks if hit]


def predict_risk(r):
    factors = risk_factors(r)
    if RISK_ML:
        x = scaler.transform(pd.DataFrame([r])[RISK_COLS].astype(float).values)
        level = LEVELS[int(np.ravel(risk_model.predict(x))[0])]
        return level, "ai", factors
    score = len(factors)
    return ("Low" if score <= 1 else "Moderate" if score == 2 else "High"), "rules", factors


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        raw = request.get_json(force=True)
        d = {k: float(raw[k]) for k in ["age", "weight", "height", "max_bpm", "avg_bpm", "resting_bpm",
                                        "duration", "fat", "water", "frequency", "experience"]}
        d["gender"], d["workout_type"] = raw["gender"], raw["workout_type"]
        if d["height"] <= 0 or d["max_bpm"] <= 0 or d["duration"] <= 0:
            raise ValueError("Height, max heart rate and duration must be above zero.")
        if d["avg_bpm"] > d["max_bpm"]:
            raise ValueError("Average heart rate cannot be higher than your maximum.")

        r, bmi_cat, intensity = build_features(d)
        calories, source = None, "ai"
        if calorie_model is not None:
            try:
                X = pd.DataFrame([r])[CALORIE_COLS].astype(float)
                calories = float(np.ravel(calorie_model.predict(X))[0])
            except Exception as e:
                print("[WARN] calorie model failed:", e)
        if calories is None:
            calories, source = formula_calories(d), "formula"

        level, engine, factors = predict_risk(r)
        rough = any(not (lo <= d[k] <= hi) for k, (lo, hi) in RANGES.items())
        return jsonify(ok=True,
                       calories=round(calories), calorie_source=source,
                       calories_per_hour=round(calories / d["duration"]),
                       risk=level, risk_engine=engine, factors=factors,
                       bmi=round(r["BMI"], 1), bmi_category=bmi_cat, intensity=intensity,
                       weekly_hours=round(r["Weekly_Workout_Hours"], 1), rough=rough)
    except (KeyError, ValueError, TypeError) as e:
        msg = str(e) if "cannot" in str(e) or "must" in str(e) else "Please fill in every field with a valid number."
        return jsonify(ok=False, error=msg), 400


if __name__ == "__main__":
    app.run(debug=True)
