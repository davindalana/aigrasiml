# gunicorn.conf.py
bind = "0.0.0.0:8080" # Pastikan ini sudah benar
workers = 2
threads = 4
timeout = 120 # Coba mulai dengan 120 detik (2 menit)
# Jika masih timeout, coba 300 detik (5 menit)
# Sesuaikan ini berdasarkan seberapa lama inferensi model Anda.
loglevel = 'info'
accesslog = '-'
errorlog = '-'