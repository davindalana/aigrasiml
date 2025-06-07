from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
import os
from pathlib import Path

app = Flask(__name__)

# ──────────────────────────────
# 1)  MUAT MODEL & SCALER DARI PATH ABSOLUT
# ──────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / 'best_irrigation_multiclass_model.keras'
SCALER_PATH = BASE_DIR / 'scaler_multiclass.pkl'

model  = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_NAMES = ['Soil_Moisture', 'Temperature', 'Air_Humidity']
message_map = {
    0: 'Tidak perlu siram',
    1: 'Perlu siram level 1',
    2: 'Perlu siram level 2',
    3: 'Perlu siram level 3'
}

# ──────────────────────────────
# 2)  ENDPOINT PREDIKSI
# ──────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # validasi input
    if not data or not all(k in data for k in FEATURE_NAMES):
        return jsonify({
            'error': 'Incomplete or invalid data. Required keys: Soil_Moisture, Temperature, Air_Humidity'
        }), 400

    df_input = pd.DataFrame([[
        data['Soil_Moisture'],
        data['Temperature'],
        data['Air_Humidity']
    ]], columns=FEATURE_NAMES)

    scaled = scaler.transform(df_input)
    probs  = model.predict(scaled, verbose=0)
    cls    = int(np.argmax(probs, axis=1)[0])

    if cls not in message_map:
        return jsonify({'error': f'Invalid class predicted: {cls}'}), 500

    return jsonify({
        'Irrigation_Level': cls,
        'Message': message_map[cls]
    })

# ──────────────────────────────
# 3)  MAIN – BIND KE HOST & PORT YG BENAR
# ──────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))   # Render meng-set PORT sendiri
    app.run(host='0.0.0.0', port=port)
