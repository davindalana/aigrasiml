import joblib

try:
    # Pastikan nama file scaler sudah benar
    scaler = joblib.load('scaler_multiclass.pkl') 
    
    # Cek jumlah fitur yang diharapkan scaler
    if hasattr(scaler, 'n_features_in_'):
        print(f"Scaler ini dilatih dengan {scaler.n_features_in_} fitur.")
    
    # Cek nama fitur yang diharapkan (jika tersedia)
    if hasattr(scaler, 'feature_names_in_'):
        print("Nama fitur yang diharapkan oleh scaler:")
        print(scaler.feature_names_in_)
        
except Exception as e:
    print(f"Gagal memuat scaler: {e}")