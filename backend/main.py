from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from PIL import Image
import numpy as np
import io

app = FastAPI()

class AdRequest(BaseModel):
    image_url: str
    campaign_id: str

def analyze_image_properties(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    img_array = np.array(img)
    brightness = float(np.mean(img_array))
    
    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 2),
        "brightness_score": round(brightness, 2),
        "format": img.format
    }

@app.get("/health")
def health_check():
    return {"status": "operational", "service": "ad-vision-v1"}

@app.post("/analyze")
async def analyze_ad(request: AdRequest):
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(request.image_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Error descargando imagen")
            
        data = analyze_image_properties(resp.content)
        
        # Regla de negocio simple
        warnings = []
        if data['brightness_score'] < 50:
            warnings.append("Low Brightness: Ad might be too dark.")

        return {
            "campaign_id": request.campaign_id,
            "analysis": data,
            "warnings": warnings
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))