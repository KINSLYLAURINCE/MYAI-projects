import os
import io
import threading
import time
import numpy as np
from PIL import Image

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cats_dogs_model.keras')
IMG_SIZE   = (160, 160)
PORT       = int(os.environ.get("PORT", 5000))

app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Upload an image to classify it as Cat or Dog.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "model":        None,
    "model_loaded": False,
    "status":       "Model not loaded yet."
}

def load_model():
    try:
        from tensorflow import keras
        print(f"[API] Loading model from {MODEL_PATH} ...")
        state["model"]        = keras.models.load_model(MODEL_PATH)
        state["model_loaded"] = True
        state["status"]       = "Model loaded and ready."
        print("[API] Model loaded successfully.")
    except Exception as e:
        state["model_loaded"] = False
        state["status"]       = f"Failed to load model: {str(e)}"
        print(f"[API] ERROR: {e}")

def self_ping():
    url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:5000") + "/health"
    while True:
        time.sleep(10 * 60)
        try:
            requests.get(url, timeout=10)
        except Exception:
            pass

@app.on_event("startup")
def startup():
    load_model()
    threading.Thread(target=self_ping, daemon=True).start()

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/health")
def health():
    return {
        "model_loaded": state["model_loaded"],
        "status":       state["status"],
        "model_path":   MODEL_PATH,
        "file_exists":  os.path.exists(MODEL_PATH),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model not ready. Check /health.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image. Got: {file.content_type}")

    image_bytes = await file.read()
    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    prediction = float(state["model"].predict(tensor, verbose=0)[0][0])
    label      = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if label == "Dog" else 1 - prediction

    return JSONResponse({
        "prediction":     label,
        "confidence":     f"{confidence * 100:.2f}%",
        "raw_score":      round(prediction, 6),
        "interpretation": "score > 0.5 = Dog, score < 0.5 = Cat"
    })

@app.post("/reload")
def reload_model():
    load_model()
    return {"status": state["status"], "loaded": state["model_loaded"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=False)
