import os
import io
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'cats_dogs_model.tflite')
IMG_SIZE    = (160, 160)
PORT        = int(os.environ.get("PORT", 5000))

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
    "interpreter": None,
    "input_index":  None,
    "output_index": None,
    "model_loaded": False,
    "status":       "Model not loaded yet."
}

# ── Load TFLite model ─────────────────────────────────────────────────────────
def load_model():
    try:
        print(f"[API] Loading TFLite model from {MODEL_PATH} ...")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        state["interpreter"]  = interpreter
        state["input_index"]  = interpreter.get_input_details()[0]["index"]
        state["output_index"] = interpreter.get_output_details()[0]["index"]
        state["model_loaded"] = True
        state["status"]       = "Model loaded and ready."
        print("[API] TFLite model loaded successfully.")
    except Exception as e:
        state["model_loaded"] = False
        state["status"]       = f"Failed to load model: {str(e)}"
        print(f"[API] ERROR: {e}")

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    load_model()

# ── Preprocessor ─────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

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
        raise HTTPException(status_code=503, detail="Model not ready. Check /health.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image. Got: {file.content_type}")

    image_bytes = await file.read()
    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    interp = state["interpreter"]
    interp.set_tensor(state["input_index"], tensor)
    interp.invoke()
    prediction = float(interp.get_tensor(state["output_index"])[0][0])

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

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=False)
