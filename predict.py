# predict.py (Revisi)
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
import os
from pathlib import Path

app = Flask(__name__)

# ──────────────────────────────
# 1) MUAT MODEL & SCALER DARI PATH ABSOLUT DENGAN PENANGANAN ERROR
# ──────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / 'best_irrigation_multiclass_model.keras'
SCALER_PATH = BASE_DIR / 'scaler_multiclass.pkl'

# Inisialisasi variabel global untuk model dan scaler
model = None
scaler = None

try:
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"INFO: Model loaded from {MODEL_PATH}")
    print(f"INFO: Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load model or scaler: {e}")
    # Jika gagal dimuat, kita bisa mengembalikan error di endpoint predict
    # atau memilih untuk menghentikan aplikasi jika ini kritis.
    # Untuk deployment, lebih baik biarkan aplikasi hidup dan kembalikan 500
    # jika model tidak tersedia saat request.

FEATURE_NAMES = ['Soil_Moisture', 'Temperature', 'Air_Humidity']
message_map = {
    0: 'Tidak perlu siram',
    1: 'Perlu siram Sedikit',
    2: 'Perlu siram Sedang',
    3: 'Perlu siram Banyak'
}

# ──────────────────────────────
# 2) ENDPOINT UTAMA & HEALTH CHECK
# ──────────────────────────────
@app.route('/', methods=['GET'])
def health_check():
    # Endpoint ini sangat penting untuk health check Railway.
    # Pastikan responsif dan tidak bergantung pada model yang dimuat.
    return jsonify(
        status="API is running",
        model_loaded=bool(model), # Informasi apakah model berhasil dimuat
        scaler_loaded=bool(scaler)
    ), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah model dan scaler berhasil dimuat
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model or scaler not loaded. API is currently unavailable.'
        }), 503 # Service Unavailable

    data = request.get_json()

    # validasi input
    if not data or not all(k in data for k in FEATURE_NAMES):
        return jsonify({
            'error': 'Incomplete or invalid data. Required keys: Soil_Moisture, Temperature, Air_Humidity'
        }), 400

    try:
        df_input = pd.DataFrame([[
            data['Soil_Moisture'],
            data['Temperature'],
            data['Air_Humidity']
        ]], columns=FEATURE_NAMES)

        scaled = scaler.transform(df_input)
        # verbose=0 untuk mengurangi output log TensorFlow saat inferensi
        probs  = model.predict(scaled, verbose=0)
        cls    = int(np.argmax(probs, axis=1)[0])

        if cls not in message_map:
            # Ini bisa terjadi jika model memprediksi di luar ekspektasi
            return jsonify({'error': f'Invalid class predicted: {cls}. Contact support.'}), 500

        return jsonify({
            'Irrigation_Level': cls,
            'Message': message_map[cls]
        })
    except Exception as e:
        # Menangkap error tak terduga selama prediksi
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

# ──────────────────────────────
# 3) MAIN – BIND KE HOST & PORT YG BENAR
# ──────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080)) # Menggunakan os.getenv secara langsung
    print(f"INFO: Starting Flask app on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)