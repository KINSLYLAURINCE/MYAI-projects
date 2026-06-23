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
MODEL_PATH = 'H:/AI/project 17/cats_dogs_model.keras'
IMG_SIZE   = (160, 160)
WATCH_INTERVAL = 10   # seconds between checks when waiting for model

# ── App + shared state ────────────────────────────────────────────────────────
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
    "model_modified": None,   # last-modified timestamp of the file
    "status":         "Model not loaded yet — training may still be in progress."
}

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    """Load (or reload) the model from disk."""
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
    """
    Runs in a background thread.
    - If model file doesn't exist yet: keep checking every WATCH_INTERVAL seconds.
    - If model file appears or is updated (training saved a better checkpoint):
      load it automatically.
    """
    print("[Watcher] Started — watching for model file...")
    while True:
        if os.path.exists(MODEL_PATH):
            current_mtime = os.path.getmtime(MODEL_PATH)

            if not state["model_loaded"]:
                # File just appeared for the first time
                print("[Watcher] Model file detected — loading...")
                load_model()

            elif current_mtime != state["model_modified"]:
                # File was updated (ModelCheckpoint saved a better version)
                print("[Watcher] Model file updated — reloading with new weights...")
                load_model()
        else:
            state["status"] = "Waiting for training to finish — model file not found yet."
            print(f"[Watcher] Model not found, checking again in {WATCH_INTERVAL}s...")

        time.sleep(WATCH_INTERVAL)

# ── Image preprocessor ────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    """Resize and normalize image exactly as done during training."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)

# ── Startup: try loading immediately, then start watcher ─────────────────────
@app.on_event("startup")
def startup():
    if os.path.exists(MODEL_PATH):
        load_model()
    else:
        print("[API] Model file not found — training may still be running.")
        print(f"[API] Will auto-load when {MODEL_PATH} appears.")

    watcher_thread = threading.Thread(target=model_watcher, daemon=True)
    watcher_thread.start()

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check whether the model is loaded and ready."""
    return {
        "model_loaded": state["model_loaded"],
        "status":       state["status"],
        "model_path":   MODEL_PATH,
        "file_exists":  os.path.exists(MODEL_PATH),
        "tip": "POST an image to /predict once model_loaded is true."
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a cat or dog image (JPG/PNG).
    Returns the predicted class and confidence score.
    """
    if not state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready yet. Check /health for status."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}"
        )

    image_bytes = await file.read()

    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    prediction = state["model"].predict(tensor, verbose=0)[0][0]

    label      = "Dog" if prediction > 0.5 else "Cat"
    confidence = float(prediction) if label == "Dog" else float(1 - prediction)

    return JSONResponse({
        "prediction":       label,
        "confidence":       f"{confidence * 100:.2f}%",
        "raw_score":        round(float(prediction), 6),
        "interpretation":   "score > 0.5 = Dog, score < 0.5 = Cat"
    })


@app.post("/reload")
def reload_model():
    """Manually force the API to reload the model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model file not found on disk.")
    load_model()
    return {
        "status":  state["status"],
        "loaded":  state["model_loaded"]
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=False)
