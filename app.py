from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import cv2
import numpy as np
import shutil
import os
from ultralytics import YOLO
from jinja2 import Template
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static folder for serving CSS & JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLOv8 model (CPU only)
model_path = "model/yolov8n.pt"
model = YOLO(model_path)
model.to("cpu")

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

    # Run YOLOv8 inference
    results = model(image)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf)
            })

    return JSONResponse(content={"detections": detections, "file_path": file_path})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
