import io
import numpy as np
import torch
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pipeline import run_pipeline

app = FastAPI(title="Marine Oil-Spill Intelligence API")

# Allows a frontend running on a different port/domain to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend URL before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_uploaded_image(file_bytes, patch_size=256):
    """
    Converts an uploaded image file into the same tensor format the model
    expects: grayscale, center-cropped/resized to patch_size, normalized.
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = image.resize((patch_size, patch_size))
    arr = np.array(image, dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return torch.from_numpy(arr).unsqueeze(0).float()  # (1, H, W)


@app.get("/")
def root():
    return {"status": "ok", "message": "Marine Oil-Spill Intelligence API is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload a SAR image. Runs it through the full pipeline:
    detection -> characterization -> drift estimate -> AIS correlation -> ranking.

    NOTE: geotransform and AIS data are currently EXAMPLE/FAKE values (see
    pipeline.py) since real geolocation + real AIS integration aren't wired
    up yet. Results are illustrative of the pipeline mechanics, not yet a
    real analysis of the uploaded image's true location.
    """
    file_bytes = await file.read()
    image_tensor = preprocess_uploaded_image(file_bytes)

    fake_geotransform = (-94.99, 0.0001, 0, 29.69, 0, -0.0001)

    result = run_pipeline(
        image_tensor=image_tensor,
        event_time=pd.Timestamp.now(),
        geotransform=fake_geotransform,
        wind_speed_ms=6.0, wind_dir_deg=180,
        current_speed_ms=0.3, current_dir_deg=90,
        hours_since_release=12,
        device=device,
    )

    if result is None:
        return {"spill_detected": False}

    return result