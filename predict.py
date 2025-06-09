from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
import os
from pathlib import Path
import requests

app = Flask(__name__)

# ───────────── Fungsi Cuaca dari OpenWeatherMap ─────────────
def get_weather_forecast():
    API_KEY = "1abea4adf5a8e3217023e324e339b83e"
    location = "Malang"
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            forecast = data['list'][0]

            will_rain = False
            rain_amount = 0.0

            if 'weather' in forecast and forecast['weather']:
                weather_condition = forecast['weather'][0]['main'].lower()
                if 'rain' in weather_condition:
                    will_rain = True
                    if 'rain' in forecast and '3h' in forecast['rain']:
                        rain_amount = forecast['rain']['3h']

            return {
                'rain_forecast': 1 if will_rain else 0,
                'rain_amount': rain_amount,
                'api_temperature': forecast['main']['temp'],
                'api_humidity': forecast['main']['humidity']
            }
    except Exception as e:
        print(f"Error fetching weather data: {e}")

    return {
        'rain_forecast': 0,
        'rain_amount': 0.0,
        'api_temperature': None,
        'api_humidity': None
    }

# ───────────── Fungsi Penyesuaian Berdasarkan Cuaca ─────────────
def calculate_weather_influence_multiclass(weather_data, sensor_data):
    adjustments = np.zeros(4)

    # Pengaruh hujan
    if weather_data.get('rain_forecast') == 1:
        rain_factor = min(weather_data.get('rain_amount', 0) / 10.0, 1.0)
        adjustments[0] += rain_factor * 0.3
        adjustments[1] += rain_factor * 0.1
        adjustments[2] -= rain_factor * 0.2
        adjustments[3] -= rain_factor * 0.4

    # Pengaruh suhu
    if weather_data.get('api_temperature') is not None:
        temp_diff = weather_data['api_temperature'] - sensor_data['temperature']
        if temp_diff > 0:
            temp_factor = min(temp_diff / 10.0, 0.3)
            adjustments[0] -= temp_factor * 0.2
            adjustments[1] -= temp_factor * 0.1
            adjustments[2] += temp_factor * 0.1
            adjustments[3] += temp_factor * 0.2

    # Pengaruh kelembaban
    if weather_data.get('api_humidity') is not None:
        humidity_diff = sensor_data['humidity'] - weather_data['api_humidity']
        if humidity_diff > 0:
            humidity_factor = min(humidity_diff / 20.0, 0.2)
            adjustments[0] -= humidity_factor * 0.1
            adjustments[1] -= humidity_factor * 0.05
            adjustments[2] += humidity_factor * 0.05
            adjustments[3] += humidity_factor * 0.1

    return adjustments

# ───────────── Load Model dan Scaler ─────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'best_irrigation_multiclass_model.keras'
SCALER_PATH = BASE_DIR / 'scaler_multiclass.pkl'

model = None
scaler = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"INFO: Model loaded from {MODEL_PATH}")
    print(f"INFO: Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load model or scaler: {e}")

# ───────────── Konstanta ─────────────
FEATURE_NAMES = ['Soil_Moisture', 'Temperature', 'Air_Humidity']
message_map = {
    0: 'Tidak perlu siram',
    1: 'Perlu siram Sedikit',
    2: 'Perlu siram Sedang',
    3: 'Perlu siram Banyak'
}

# ───────────── Endpoint Health Check ─────────────
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'API is running',
        'model_loaded': bool(model),
        'scaler_loaded': bool(scaler)
    }), 200

# ───────────── Endpoint Prediksi ─────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 503
    

    data = request.get_json()
    if not data or not all(k in data for k in FEATURE_NAMES):
        return jsonify({'error': 'Missing input. Required: Soil_Moisture, Temperature, Air_Humidity'}), 400

    try:
        df_input = pd.DataFrame([[
            data['Soil_Moisture'],
            data['Temperature'],
            data['Air_Humidity']
        ]], columns=FEATURE_NAMES)

        scaled = scaler.transform(df_input)
        probs = model.predict(scaled, verbose=0)[0]

        weather_data = get_weather_forecast()
        if weather_data:
            sensor_data = {
                'temperature': data['Temperature'],
                'humidity': data['Air_Humidity']
            }
            adjustments = calculate_weather_influence_multiclass(weather_data, sensor_data)
            probs += adjustments
            probs = np.clip(probs, 0, None)
            probs = probs / probs.sum()
            
        cls = int(np.argmax(probs))

        return jsonify({
            'Irrigation_Level': cls,
            'Message': message_map[cls]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ───────────── Jalankan Aplikasi ─────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # default 8080 jika tidak diset
    print(f"INFO: Starting Flask app on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
