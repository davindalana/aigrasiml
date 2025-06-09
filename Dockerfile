FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy semua kode & model ke dalam container
COPY . .

# Jalankan dengan gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "predict:app"]