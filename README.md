# Ad-Vision Intelligence Engine 

> Motor de auditoría automatizada para creatividades publicitarias (Banners/Ads) basado en Microservicios, Computer Vision y OCR Multilingüe.

## Descripción del Proyecto
Este sistema permite analizar imágenes publicitarias en tiempo real para extraer métricas técnicas y semánticas. Utiliza una arquitectura contenerizada para orquestar modelos de Inteligencia Artificial que validan si un anuncio cumple con los estándares de calidad antes de salir al aire.

**Ideal para:** AdTech QA, validación de Brand Safety y análisis de competencia.

##  Key Features

* **Detección de Objetos (YOLOv8):** Identifica elementos visuales clave (personas, productos, vehículos) para asegurar la relevancia del anuncio.
* **OCR Multilingüe Robusto:** Lectura de texto en **Portugués (PT), Español (ES) e Inglés (EN)**.
* **Visión Adaptativa:**
    * *Smart Upscaling:* Agranda imágenes de baja resolución automáticamente.
    * *Otsu Thresholding:* Algoritmo de visión artificial para leer texto sobre fondos complejos (degradados, bajo contraste).
* **Métricas Técnicas:** Cálculo de brillo promedio y dimensiones para validar formatos (Mobile/Desktop).
* **Arquitectura Resiliente:** API Gateway (Nginx) + Backend Asíncrono (FastAPI) + Docker.

## Stack Tecnológico

* **Infraestructura:** Docker & Docker Compose.
* **Gateway:** Nginx (Reverse Proxy).
* **Backend:** Python 3.9, FastAPI, Uvicorn.
* **AI & Vision:**
    * `Ultralytics YOLOv8` (Object Detection).
    * `Tesseract 5` (OCR Engine).
    * `OpenCV` (Image Pre-processing).
    * `NumPy` & `Pillow`.

## Instalación y Ejecución

El proyecto está dockerizado, por lo que no requiere instalar Python ni librerías en el host.

### Prerrequisitos
* Docker Desktop instalado.

### Pasos
1.  Clonar el repositorio:
    ```bash
    git clone [https://github.com/AndresGM7/ad-vision-gateway.git](https://github.com/AndresGM7/ad-vision-gateway.git)
    cd ad-vision-gateway
    ```

2.  Levantar la arquitectura:
    ```bash
    docker compose up --build
    ```

3.  Verificar estado:
    * Healthcheck: `http://localhost:8080/health`

## Uso de la API

### Endpoint: `/analyze` [POST]

Envía una URL de una imagen pública para ser auditada.

**Ejemplo (cURL):**
```bash
curl -X POST "http://localhost:8080/analyze" \
     -H "Content-Type: application/json" \
     -d '{
           "campaign_id": "DEMO_Q1", 
           "image_url": "[https://upload.wikimedia.org/wikipedia/commons/f/f3/Placa_Pare_Brasil.jpeg](https://upload.wikimedia.org/wikipedia/commons/f/f3/Placa_Pare_Brasil.jpeg)"
         }'