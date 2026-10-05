"""FitCoach: CatBoost fitness classification + XGBoost calorie regression.
Export the two trained bundles using export_models.py before predicting.
"""
from pathlib import Path
import math
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024
BUNDLES = {}
LOAD_ERRORS = {}

# key, display label, minimum, maximum, step, default, dataset column
FIELDS = [
    ('age', 'Age (years)', 18, 100, 1, 25, 'Age'),
    ('weight', 'Weight (kg)', 25, 300, .1, 75, 'Weight (kg)'),
    ('height', 'Height (m)', 1, 2.5, .01, 1.75, 'Height (m)'),
    ('fat', 'Body fat (%)', 1, 65, .1, 22, 'Fat_Percentage'),
    ('max_bpm', 'Maximum heart rate (BPM)', 60, 250, 1, 180, 'Max_BPM'),
    ('avg_bpm', 'Average heart rate (BPM)', 35, 230, 1, 140, 'Avg_BPM'),
    ('resting_bpm', 'Resting heart rate (BPM)', 30, 150, 1, 60, 'Resting_BPM'),
    ('duration', 'Session duration (hours)', .1, 5, .01, 1, 'Session_Duration (hours)'),
    ('water', 'Daily water intake (liters)', .1, 8, .1, 2.5, 'Water_Intake (liters)'),
    ('frequency', 'Workouts per week', 1, 7, 1, 4, 'Workout_Frequency (days/week)'),
]
OPTIONS = {
    'gender': ('Gender', ['Male', 'Female'], 'Gender'),
    'workout': ('Workout type', ['Cardio', 'Strength', 'HIIT', 'Yoga'], 'Workout_Type'),
    'experience': ('Experience level', ['1', '2', '3'], 'Experience_Level'),
}
LABELS = {0: 'Low', 1: 'Moderate', 2: 'High'}


def load_models():
    """Load only locally exported, trusted pickle files; never accept uploads."""
    for task in ('classification', 'regression'):
        path = BASE_DIR / 'models' / f'{task}_bundle.pkl'
        try:
            with path.open('rb') as file:
                bundle = pickle.load(file)
            required = {'model', 'encoder', 'categorical_columns', 'feature_columns', 'task'}
            if not isinstance(bundle, dict) or not required.issubset(bundle):
                raise ValueError('Invalid model bundle. Export it again.')
            if bundle['task'] != task:
                raise ValueError('The model bundle belongs to the wrong task.')
            BUNDLES[task] = bundle
        except FileNotFoundError:
            LOAD_ERRORS[task] = f'Missing models/{task}_bundle.pkl. Follow README export steps, then restart.'
        except Exception:
            app.logger.exception('Cannot load %s bundle', task)
            LOAD_ERRORS[task] = f'Cannot load {task} model. Use the same package versions as your notebook and export again.'


def make_features(payload):
    if not isinstance(payload, dict):
        raise ValueError('Send the workout inputs as a JSON object.')
    data = {}
    for key, label, minimum, maximum, step, default, column in FIELDS:
        value = payload.get(key)
        if isinstance(value, bool):
            raise ValueError(f'{label}: enter a number.')
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{label}: enter a valid number.') from None
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f'{label}: enter a value from {minimum} to {maximum}.')
        if step == 1 and not number.is_integer():
            raise ValueError(f'{label}: enter a whole number.')
        data[column] = int(number) if step == 1 else number
    for key, (label, choices, column) in OPTIONS.items():
        value = str(payload.get(key, ''))
        if value not in choices:
            raise ValueError(f'{label}: select one of the available options.')
        data[column] = int(value) if key == 'experience' else value
    if not data['Resting_BPM'] <= data['Avg_BPM'] <= data['Max_BPM']:
        raise ValueError('Heart rates must follow: resting ≤ average ≤ maximum.')
    bmi = data['Weight (kg)'] / data['Height (m)'] ** 2
    intensity = data['Avg_BPM'] / data['Max_BPM']
    data.update({
        'BMI': bmi,
        'BMI_Category': ('Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'),
        'Heart_Rate_Reserve': data['Max_BPM'] - data['Resting_BPM'],
        'HR_Intensity': intensity,
        'Workout_Load': data['Session_Duration (hours)'] * data['Avg_BPM'],
        'Weekly_Workout_Hours': data['Workout_Frequency (days/week)'] * data['Session_Duration (hours)'],
        'Workout_Intensity': 'Low' if intensity < .70 else 'Moderate' if intensity < .85 else 'High',
    })
    return pd.DataFrame([data])


def transform_features(frame, bundle):
    """Reproduce notebook encoding and enforce the exact training column order."""
    columns = bundle['categorical_columns']
    encoder = bundle['encoder']
    # Refuse unseen categories instead of silently mapping them to a baseline.
    for column, categories in zip(columns, encoder.categories_):
        if frame.iloc[0][column] not in categories:
            raise ValueError(f'{column}: this category was not present during training.')
    values = encoder.transform(frame[columns])
    if hasattr(values, 'toarray'):
        values = values.toarray()
    encoded = pd.DataFrame(values, columns=encoder.get_feature_names_out(columns), index=frame.index)
    features = pd.concat([frame.drop(columns=columns), encoded], axis=1)
    missing = set(bundle['feature_columns']) - set(features.columns)
    if missing:
        raise RuntimeError('Model feature schema does not match this app. Export models again.')
    return features.loc[:, bundle['feature_columns']]


@app.get('/')
def index():
    return render_template('index.html', fields=FIELDS, options=OPTIONS, ready=len(BUNDLES) == 2,
                           setup_errors=list(LOAD_ERRORS.values()))


@app.post('/predict')
def predict():
    if len(BUNDLES) != 2:
        return jsonify(error='Models are not ready. Export both model bundles and restart the app.'), 503
    try:
        frame = make_features(request.get_json(silent=True))
        classifier = BUNDLES['classification']
        regressor = BUNDLES['regression']
        cx = transform_features(frame, classifier)
        rx = transform_features(frame, regressor)
        class_value = int(np.asarray(classifier['model'].predict(cx)).ravel()[0])
        risk = LABELS[class_value]
        probabilities = np.asarray(classifier['model'].predict_proba(cx))[0]
        classes = np.asarray(classifier['model'].classes_).ravel()
        probability_items = [{'label': LABELS[int(label)], 'value': round(float(p) * 100, 1)}
                             for label, p in zip(classes, probabilities)]
        calories = float(np.asarray(regressor['model'].predict(rx)).ravel()[0])
        if not math.isfinite(calories) or calories < 0:
            raise RuntimeError('Model returned an invalid calorie estimate.')
        row = frame.iloc[0]
        score = sum([row['BMI'] >= 30, row['Fat_Percentage'] >= 30, row['HR_Intensity'] >= .85,
                     row['Resting_BPM'] >= 70, row['Workout_Load'] >= 200])
        rule_risk = 'Low' if score <= 1 else 'Moderate' if score == 2 else 'High'
        return jsonify(risk=risk, calories=round(calories), probabilities=probability_items,
                       bmi=round(float(row['BMI']), 1), bmi_category=row['BMI_Category'],
                       intensity=row['Workout_Intensity'], intensity_percent=round(float(row['HR_Intensity']) * 100, 1),
                       weekly_hours=round(float(row['Weekly_Workout_Hours']), 2),
                       risk_score=int(score), rule_risk=rule_risk,
                       classification_model=classifier['model_name'], regression_model=regressor['model_name'])
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify(error='Prediction failed. Check the terminal and re-export compatible model bundles.'), 500


load_models()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)), debug=False)
