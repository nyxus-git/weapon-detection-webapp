import sys
sys.path.append("./yolov5")
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import cv2
import numpy as np
import shutil
import os
from jinja2 import Template
from fastapi.staticfiles import StaticFiles
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords

# Initialize FastAPI
app = FastAPI()

# Mount static folder for serving CSS & JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLOv5 model (CPU mode)
model_path = "best.pt"
device = select_device("cpu")
model = DetectMultiBackend(model_path, device=device, dnn=False)

# Upload folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    with open("templates/index.html") as f:
        html_content = Template(f.read()).render()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read image
    image = cv2.imread(file_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize

    # Run YOLOv5 inference
    pred = model(img)
    detections = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

    # Extract detection results
    results = []
    for det in detections:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in det:
                results.append({
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "bbox": [int(coord) for coord in xyxy]
                })

    return JSONResponse(content={"detections": results, "file_path": file_path})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
