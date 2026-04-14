from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import os

app = FastAPI(title="WasteAI — Garbage Classification API")
logging.basicConfig(level=logging.INFO)

templates = Jinja2Templates(directory="templates")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

model = None

CLASS_NAMES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model("model/final_model.h5", compile=False)
        logging.info("Model loaded")
    except Exception as e:
        logging.error(f"Error loading: {e}")
        raise RuntimeError("Model failed to load")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        return np.expand_dims(arr, axis=0)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file")

def run_inference(image_bytes: bytes) -> dict:
    arr = preprocess_image(image_bytes)
    preds = model.predict(arr, verbose=0)

    top3_idx = np.argsort(preds[0])[-3:][::-1]
    top3 = [
        {"class": CLASS_NAMES[i], "confidence": round(float(preds[0][i]), 4)}
        for i in top3_idx
    ]

    return {
        "top_prediction": top3[0],
        "top_3_predictions": top3
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()

    try:
        result = run_inference(image_bytes)
        logging.info(f"Predicted: {result['top_prediction']}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed — please try another image")