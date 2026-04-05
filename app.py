import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Preload Models from disk
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'best_rul_model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model, scaler = None, None

# Expected by scaler in exact order:
SCALER_FEATURES = [
    'op_setting_1', 'op_setting_2', 'op_setting_3', 'sensor_1', 'sensor_2', 
    'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 
    'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 
    'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
]

# Required by model (the ones that had sufficient variance)
MODEL_FEATURES = [
    'op_setting_1', 'op_setting_2', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 
    'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 
    'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', features=MODEL_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return "Models are not loaded correctly. Please check server logs.", 500
        
    try:
        # Get user inputs for model features
        user_inputs = {}
        for feature in MODEL_FEATURES:
            val = request.form.get(feature)
            user_inputs[feature] = float(val) if val else 0.0

        # Build the 24-feature array for scaler (default to 0.0 for dropped features)
        full_array = []
        for sf in SCALER_FEATURES:
            full_array.append(user_inputs.get(sf, 0.0))
            
        full_df = pd.DataFrame([full_array], columns=SCALER_FEATURES)
        
        # Scale all
        scaled_array = scaler.transform(full_df)
        
        # Extract just the model features from scaled output
        scaled_df = pd.DataFrame(scaled_array, columns=SCALER_FEATURES)
        model_df = scaled_df[MODEL_FEATURES]
        
        # Predict RUL
        prediction_raw = model.predict(model_df)
        
        # Cap logic: clip between 0 and 125 based on training constraints
        prediction = max(0.0, min(125.0, round(float(prediction_raw[0]), 1)))
        
        return render_template('index.html', features=MODEL_FEATURES, prediction=prediction)
    except Exception as e:
        return render_template('index.html', features=MODEL_FEATURES, error=str(e))

if __name__ == '__main__':
    # Run the Flask app on localhost
    app.run(debug=True, host='127.0.0.1', port=5000)
