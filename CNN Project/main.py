from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

app = FastAPI(title="Garbage Classification API")
logging.basicConfig(level=logging.INFO)
model = None

class_names = [
    'battery','biological','cardboard','clothes','glass',
    'metal','paper','plastic','shoes','trash'
]

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(
            "model/best_model.keras",
            compile=False
        )
        logging.info("Model loaded")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise RuntimeError("Model failed to load")




@app.get("/")
def read_root():
    return {"message": "running"}

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

def predict_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class, confidence = predict_image(image_bytes)

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4)
    }