from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib

app = Flask(__name__)

# Load model dan scaler
model = tf.keras.models.load_model('best_irrigation_multiclass_model.keras')
scaler = joblib.load('scaler_multiclass.pkl')

# Nama fitur harus sama dengan saat scaler dilatih
FEATURE_NAMES = ['Soil_Moisture', 'Temperature', 'Air_Humidity']

# Mapping untuk pesan berdasarkan Irrigation_Level
message_map = {
    0: 'Tidak perlu siram',
    1: 'Perlu siram level 1',
    2: 'Perlu siram level 2',
    3: 'Perlu siram level 3'
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validasi input
        if not all(k in data for k in FEATURE_NAMES):
            return jsonify({'error': 'Incomplete or invalid data. Required keys: Soil_Moisture, Temperature, Air_Humidity'}), 400

        # Masukkan ke DataFrame agar sesuai dengan scaler
        df_input = pd.DataFrame([[
            data['Soil_Moisture'],
            data['Temperature'],
            data['Air_Humidity']
        ]], columns=FEATURE_NAMES)

        # Skalakan input
        scaled_input = scaler.transform(df_input)

        # Prediksi
        prediction = model.predict(scaled_input, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Validasi kelas
        if predicted_class not in [0, 1, 2, 3]:
            return jsonify({'error': f'Invalid class predicted: {predicted_class}'}), 500

        # Dapatkan pesan berdasarkan kelas
        message = message_map.get(predicted_class, 'Kelas tidak dikenal')

        return jsonify({

            'Irrigation_Level': int(predicted_class),
            'Message': message
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8001)