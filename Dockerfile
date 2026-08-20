FROM python:3.11-slim

WORKDIR /app

# Derleme bağımlılıkları için gerekli sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# Uvicorn sunucusunu başlat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]