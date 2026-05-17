# Menggunakan image Python resmi yang ringan
FROM python:3.10-slim

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements terlebih dahulu agar bisa di-cache oleh Docker
COPY requirements.txt .

# Menginstal semua dependensi proyek
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh core program ke dalam container
COPY . .

# Menjalankan gunicorn sebagai server production (menangkap PORT dinamis dari Cloud Run)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app