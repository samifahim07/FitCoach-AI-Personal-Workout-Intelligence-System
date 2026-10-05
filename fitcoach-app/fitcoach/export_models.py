"""Run INSIDE each completed notebook using %run -i (see README).
Classification: %run -i export_models.py classification
Regression:     %run -i export_models.py regression
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def export_bundle(task, namespace):
    if task not in ('classification', 'regression'):
        raise ValueError('Choose classification or regression.')
    required = ['df', 'xtrain', 'model_cat' if task == 'classification' else 'model_xgb']
    missing = [name for name in required if name not in namespace]
    if missing:
        raise ValueError('Run your notebook first. Missing variables: ' + ', '.join(missing))
    df = namespace['df']
    training = namespace['xtrain']
    if task == 'classification':
        raw = df.drop(columns=['Fitness_Risk', 'Risk_Score', 'Calories_Burned', 'Calories_Per_Hour'])
        cats = list(raw.select_dtypes(include='object').columns)
        # Original notebook refits a single encoder for each column. Recreate
        # the same category ordering and drop-first encoding in one encoder.
        saved_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='error')
        saved_encoder.fit(raw[cats])
        model = namespace['model_cat']
        model_name = 'CatBoost Classifier'
    else:
        raw = df.drop(columns=['Calories_Burned', 'Calories_Per_Hour', 'BMI_Category', 'Risk_Score', 'Fitness_Risk'])
        if 'encoder' not in namespace:
            raise ValueError('Run the regression encoding cell first.')
        saved_encoder = namespace['encoder']
        cats = list(saved_encoder.feature_names_in_)
        model = namespace['model_xgb']
        model_name = 'XGBoost Regressor'
    # Check export against actual notebook training values, not just column names.
    original = raw.loc[training.index]
    encoded = pd.DataFrame(saved_encoder.transform(original[cats]),
                           columns=saved_encoder.get_feature_names_out(cats), index=training.index)
    reconstructed = pd.concat([original.drop(columns=cats), encoded], axis=1)
    reconstructed = reconstructed.loc[:, training.columns]
    if not np.allclose(reconstructed.to_numpy(dtype=float), training.to_numpy(dtype=float), equal_nan=True):
        raise ValueError('Encoding differs from xtrain. Re-run the original notebook cells before exporting.')
    # Ensure the selected estimator is already fitted and can accept this schema.
    model.predict(reconstructed.iloc[:1])
    fitted_model = getattr(model, 'best_estimator_', model)
    bundle = {'task': task, 'model': fitted_model, 'model_name': model_name,
              'encoder': saved_encoder, 'categorical_columns': cats,
              'feature_columns': list(training.columns)}
    folder = Path('models')
    folder.mkdir(exist_ok=True)
    destination = folder / f'{task}_bundle.pkl'
    with destination.open('wb') as file:
        pickle.dump(bundle, file)
    print(f'Saved: {destination.resolve()}')
    print('Copy this file into the Flask project models folder. Restart app.py.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Use: %run -i export_models.py classification (or regression)')
    export_bundle(sys.argv[1], globals())
