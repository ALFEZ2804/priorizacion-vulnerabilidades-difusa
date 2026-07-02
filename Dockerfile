# Imagen autocontenida para evaluar el sistema sin instalación local.
#
#   docker build -t fuzzy-vuln .
#   docker run -p 8000:8000 fuzzy-vuln      ->  http://localhost:8000
#
# Los datos ya procesados van incluidos en la imagen, así que la aplicación
# arranca sin descargar nada de las APIs ni configurar ninguna clave.
FROM python:3.11-slim

WORKDIR /app

# libgomp1: la necesitan scikit-learn / scipy en tiempo de ejecución
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencias primero, para aprovechar la caché de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código y datos procesados
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
