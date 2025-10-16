# main.py
import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image

# Configuração do Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="Qwen-Image API Bridge for n8n")

# Modelos de dados
class LoadModelRequest(BaseModel):
    model_type: str  # "UNET", "CLIP", "VAE"
    filename: str

class ApplyLoraRequest(BaseModel):
    model: str
    clip: str
    lora_path: str
    strength_model: float
    strength_clip: float

class AdjustSamplingRequest(BaseModel):
    model: str
    shift: float

class GenerateImageRequest(BaseModel):
    model: str
    positive: str
    negative: str
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str

class UpscaleRequest(BaseModel):
    image: str  # base64
    model: str
    factor: float

class FilterRequest(BaseModel):
    image: str
    filter: str
    intensity: float

class GeminiRequest(BaseModel):
    prompt: str
    image_b64: Optional[str] = None

# Mock: em produção, substitua por chamadas reais ao ComfyUI
def mock_b64_image(width=1248, height=896):
    img = Image.new("RGB", (width, height), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# Endpoints
@app.post("/load-model")
def load_model(req: LoadModelRequest):
    return {"id": f"{req.model_type}_{req.filename}", req.model_type.lower(): f"{req.model_type}_{req.filename}"}

@app.post("/apply-lora")
def apply_lora(req: ApplyLoraRequest):
    return {
        "model": f"{req.model}_lora_{req.lora_path}",
        "clip": f"{req.clip}_lora_{req.lora_path}"
    }

@app.post("/adjust-sampling")
def adjust_sampling(req: AdjustSamplingRequest):
    return {"model": f"{req.model}_shift_{req.shift}"}

@app.post("/generate")
def generate_image(req: GenerateImageRequest):
    return {"latent": mock_b64_image(req.width, req.height)}

@app.post("/decode")
def decode_latent(latent: str, vae: str):
    return {"image": latent}  # Simula decodificação

@app.post("/upscale")
def upscale(req: UpscaleRequest):
    return {"image": req.image}  # Retorna a mesma imagem (mock)

@app.post("/apply-filter")
def apply_filter(req: FilterRequest):
    return {"image": req.image}

@app.post("/gemini/generate-prompt")
async def gemini_generate_prompt(req: GeminiRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set")

    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    try:
        if req.image_b64:
            image_data = base64.b64decode(req.image_b64)
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            response = model.generate_content([req.prompt, pil_image])
        else:
            response = model.generate_content(req.prompt)
        return {"text": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_enabled": bool(GOOGLE_API_KEY)
    }
