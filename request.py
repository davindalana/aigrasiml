import requests
import json

# URL endpoint server Flask Anda yang sedang berjalan
url = 'http://127.0.0.1:8001/predict' 

# Data dummy untuk pengujian
# Pastikan key-nya (kunci) sama dengan yang Anda kirim dari ESP32
# yaitu: 'soil_moisture', 'temperature', 'humidity'
payload = {
    'soil_moisture': 805,
    'temperature': 30,
    'humidity': 100
}

try:
    # Mengirim request POST dengan data JSON
    response = requests.post(url, json=payload)
    
    # Mencetak status code dan response dari server
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", response.json())

except requests.exceptions.ConnectionError as e:
    print(f"Error: Tidak dapat terhubung ke server.")
    print("Pastikan server Flask (app.py) Anda sudah berjalan sebelum menjalankan skrip ini.")