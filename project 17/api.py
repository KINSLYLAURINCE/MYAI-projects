import os
import io
import time
import threading
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cats_dogs_model.keras')
IMG_SIZE   = (160, 160)
WATCH_INTERVAL = 10
PORT = int(os.environ.get("PORT", 5000))

# ── App ───────────────────────────────────────────────────────────────────────
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
    "model":          None,
    "model_loaded":   False,
    "model_modified": None,
    "status":         "Model not loaded yet."
}

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    try:
        print(f"\n[API] Loading model from {MODEL_PATH} ...")
        state["model"]          = keras.models.load_model(MODEL_PATH)
        state["model_loaded"]   = True
        state["model_modified"] = os.path.getmtime(MODEL_PATH)
        state["status"]         = "Model loaded and ready."
        print("[API] Model loaded successfully.")
    except Exception as e:
        state["model_loaded"] = False
        state["status"]       = f"Failed to load model: {str(e)}"
        print(f"[API] ERROR loading model: {e}")

# ── Background watcher ────────────────────────────────────────────────────────
def model_watcher():
    print("[Watcher] Started — watching for model file...")
    while True:
        if os.path.exists(MODEL_PATH):
            current_mtime = os.path.getmtime(MODEL_PATH)
            if not state["model_loaded"]:
                print("[Watcher] Model file detected — loading...")
                load_model()
            elif current_mtime != state["model_modified"]:
                print("[Watcher] Model file updated — reloading...")
                load_model()
        else:
            state["status"] = "Model file not found."
            print(f"[Watcher] Model not found, checking again in {WATCH_INTERVAL}s...")
        time.sleep(WATCH_INTERVAL)

# ── Image preprocessor ────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    if os.path.exists(MODEL_PATH):
        load_model()
    watcher_thread = threading.Thread(target=model_watcher, daemon=True)
    watcher_thread.start()

# ── Endpoints ─────────────────────────────────────────────────────────────────
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
        raise HTTPException(status_code=503, detail="Model is not ready yet. Check /health for status.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image. Got: {file.content_type}")

    image_bytes = await file.read()
    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    prediction = state["model"].predict(tensor, verbose=0)[0][0]
    label      = "Dog" if prediction > 0.5 else "Cat"
    confidence = float(prediction) if label == "Dog" else float(1 - prediction)

    return JSONResponse({
        "prediction":     label,
        "confidence":     f"{confidence * 100:.2f}%",
        "raw_score":      round(float(prediction), 6),
        "interpretation": "score > 0.5 = Dog, score < 0.5 = Cat"
    })

@app.post("/reload")
def reload_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model file not found on disk.")
    load_model()
    return {"status": state["status"], "loaded": state["model_loaded"]}

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=False)
