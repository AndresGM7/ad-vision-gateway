# 1. Elegimos el Sistema Operativo base (Linux con Python)
FROM python:3.9-slim

# 2. Creamos una carpeta de trabajo dentro del contenedor
WORKDIR /app

# 3. Primero copiamos solo los requerimientos (Truco de caché para que sea rápido)
COPY requirements.txt .

# 4. Instalamos las librerías dentro del contenedor
RUN pip install --no-cache-dir -r requirements.txt

# 5. AQUI ESTA LA MAGIA: Copiamos el resto del código (main.py)
COPY . .

# 6. Decimos qué comando ejecutar al arrancar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]