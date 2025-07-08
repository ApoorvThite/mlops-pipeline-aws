# src/inference.py

import os
import joblib
import pandas as pd

def model_fn(model_dir):
    """Load model from model_dir (SageMaker format)"""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type='application/json'):
    """Parse input JSON to pandas DataFrame"""
    if content_type == 'application/json':
        df = pd.read_json(request_body, orient='records')
        return df
    raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(input_data, model):
    """Make predictions"""
    return model.predict(input_data)

def output_fn(prediction, accept='application/json'):
    """Format prediction output"""
    return str(prediction.tolist())