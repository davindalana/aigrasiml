# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:{}".format(os.environ.get("PORT", "5000"))
workers = 1
threads = 1
timeout = 120       # Tambah timeout supaya model punya cukup waktu loading
preload_app = True
