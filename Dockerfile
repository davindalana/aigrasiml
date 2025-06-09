# Dockerfile
# Gunakan base image Python yang sesuai (misalnya, Python 3.9 slim-buster)
FROM python:3.11-slim-buster

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin requirements.txt dan install dependensi terlebih dahulu
# Ini memanfaatkan Docker cache layer untuk build yang lebih cepat
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek Anda ke dalam container
COPY . .

# Paparkan port yang akan didengarkan aplikasi Anda (sesuaikan jika Flask berjalan di port lain)
EXPOSE 8080

# Perintah untuk menjalankan Gunicorn
# Gunakan -c gunicorn.conf.py untuk memuat konfigurasi Gunicorn
# gunicorn <nama_modul_aplikasi>:<variabel_aplikasi_flask>
CMD ["gunicorn", "-c", "gunicorn.conf.py", "predict:app"]