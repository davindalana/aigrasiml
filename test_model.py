import numpy as np
import tensorflow as tf
import pandas as pd
import joblib

# Load model dan scaler
model = tf.keras.models.load_model('best_irrigation_multiclass_model.keras')
scaler = joblib.load('scaler_multiclass.pkl')

# Daftar input sensor
test_data = [
    [805, 30, 100],
    [425, 30, 100],
    [50, 30, 60],
    [895, 28.2, 60.1],
    [20, 25, 80],
    [90, 35, 40],
]

# Mapping untuk pesan berdasarkan Irrigation_Level
message_map = {
    0: 'Tidak perlu siram',
    1: 'Perlu siram level 1',
    2: 'Perlu siram level 2',
    3: 'Perlu siram level 3'
}

# List untuk menyimpan hasil
results = []

# Prediksi
for i, data in enumerate(test_data, 1):
    # Buat DataFrame untuk input
    df_input = pd.DataFrame([data], columns=['Soil_Moisture', 'Temperature', 'Air_Humidity'])
    
    # Skalakan input
    scaled = scaler.transform(df_input)
    
    # Prediksi
    prediction = model.predict(scaled, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Validasi kelas
    if predicted_class not in [0, 1, 2, 3]:
        print(f"Data ke-{i}: Kelas tidak valid ({predicted_class})")
        continue
    
    # Dapatkan pesan berdasarkan kelas
    message = message_map.get(predicted_class, 'Kelas tidak dikenal')
    
    # Simpan hasil ke list
    results.append({
        'Soil_Moisture': data[0],
        'Temperature': data[1],
        'Air_Humidity': data[2],
        'Irrigation_Level': predicted_class,
        'Message': message
    })
    
    # Cetak hasil (untuk debugging)
    print(f"Data ke-{i}: {data} → Prediksi Kelas: {predicted_class} → {message}")