from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from PIL import Image
import numpy as np
import io
import pytesseract
import cv2  # OpenCV
from ultralytics import YOLO

app = FastAPI()

print("--- CARGANDO CEREBRO DE IA (YOLOv8) ---")
model = YOLO('yolov8n.pt') 

class AdRequest(BaseModel):
    image_url: str
    campaign_id: str

def robust_preprocess(pil_img):
    """
    NIVEL EXPERTO: Pipeline Híbrido.
    Usa Binarización de Otsu, que es matemáticamente óptima para
    separar texto del fondo sin introducir ruido en imágenes limpias.
    """
    # 1. Convertir a OpenCV (Grises)
    img_cv = np.array(pil_img)
    if len(img_cv.shape) == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # 2. Upscaling inteligente (solo si es pequeña)
    height, width = img_cv.shape
    if width < 1000:
        scale = 2
        img_cv = cv2.resize(img_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 3. FILTRO DEFINITIVO: Binarización de Otsu
    # En lugar de adivinar el umbral, Otsu analiza el histograma de la imagen
    # y encuentra el punto exacto de separación entre texto y fondo.
    # Es mucho más limpio que el Adaptive Threshold para señales claras.
    blur = cv2.GaussianBlur(img_cv, (5,5), 0) # Suavizamos un poco antes
    _, img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invertir colores si es necesario (Tesseract prefiere texto negro sobre blanco)
    # Contamos pixeles blancos; si hay más blancos que negros, asumimos fondo blanco.
    # Si hay más negros (fondo oscuro con letras claras), invertimos.
    n_white = np.sum(img_binary == 255)
    n_black = np.sum(img_binary == 0)
    
    if n_black > n_white:
        img_binary = cv2.bitwise_not(img_binary)
    
    return Image.fromarray(img_binary)

def analyze_image_intelligence(image_bytes):
    # Cargar imagen original para métricas y YOLO
    img_original = Image.open(io.BytesIO(image_bytes))
    if img_original.mode != 'RGB':
        img_original = img_original.convert('RGB')
    
    # A. TÉCNICO
    width, height = img_original.size
    img_cv_original = np.array(img_original)
    brightness = float(np.mean(img_cv_original))
    
    # B. DETECCIÓN DE OBJETOS (YOLO usa imagen original a color)
    results = model(img_original, conf=0.25, verbose=False)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_objects.append(class_name)
    unique_objects = {i: detected_objects.count(i) for i in detected_objects}

    # C. OCR ROBUSTO
    try:
        # Usamos el preprocesador "Tanque"
        img_processed = robust_preprocess(img_original)
        
        # Guardamos la imagen procesada en logs (simulado) para debug visual si quisieras
        # img_processed.save("debug_processed.jpg") 

        # Configuración para leer TODO lo que parezca texto
        custom_config = r'--oem 3 --psm 11'
        extracted_text = pytesseract.image_to_string(
            img_processed, 
            lang='por+spa+eng', 
            config=custom_config
        )
        clean_text = extracted_text.strip().replace("\n", " ")
        error_msg = None
    except Exception as e:
        clean_text = ""
        error_msg = str(e)

    return {
        "technical": { "width": width, "height": height, "brightness": round(brightness, 2) },
        "ai_vision": { "objects_detected": unique_objects, "object_count": len(detected_objects) },
        "ai_text": {
            "raw_text": clean_text,
            "has_text": len(clean_text) > 0,
            "debug_error": error_msg
        }
    }

@app.get("/health")
def health_check():
    return {"status": "operational", "mode": "Robust Adaptive Thresholding"}

@app.post("/analyze")
async def analyze_ad(request: AdRequest):
    try:
        # HEADER SPOOFING: Nos disfrazamos de Chrome para que Wikipedia no nos bloquee
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(request.image_url, headers=headers, follow_redirects=True)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Error descargando ({resp.status_code})")
            
        analysis = analyze_image_intelligence(resp.content)
        
        warnings = []
        if not analysis['ai_text']['has_text']:
            warnings.append("Missing Copy: No text detected.")

        return {
            "campaign_id": request.campaign_id,
            "analysis": analysis,
            "warnings": warnings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))