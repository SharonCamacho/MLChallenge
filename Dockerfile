# syntax=docker/dockerfile:1.2
# put you docker configuration here
FROM python:3.11-slim

WORKDIR /app

# Instalamos dependencias primero para aprovechar el caché
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el contenido del proyecto
COPY . .

# Cloud Run inyecta la variable PORT, es mejor usarla dinámicamente
CMD uvicorn challenge.api:app --host 0.0.0.0 --port $PORT