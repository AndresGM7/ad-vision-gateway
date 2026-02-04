FROM python:3.9-slim

WORKDIR /app

# 1. INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA (System Level)
# - libgl1 y libglib2.0: Necesarios para que OpenCV funcione.
# - tesseract-ocr: El motor de lectura de texto.
# - tesseract-ocr-spa: Idioma español para el OCR.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiamos requirements e instalamos librerías de Python
COPY requirements.txt .
# --no-cache-dir reduce el tamaño de la imagen final
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiamos el código
COPY . .

# 4. Arrancamos
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]