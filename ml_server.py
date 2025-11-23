from fastapi import FastAPI, File, Form, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

model = YOLO("yolov8n.pt")


@app.post("/infer")
async def infer(
    frame: UploadFile = File(...),             # JPEG изображение
    stream_id: str = Form(...),                # ID видеопотока
    index: str = Form(...),                    # Номер кадра
    timestamp: str = Form(...),                # Временная метка
    format: str = Form(...),                   # jpeg
):
    # Проверка формата
    if format.lower() != "jpeg":
        return {"error": "unsupported format"}

    # Чтение байтов изображения
    img_bytes = await frame.read()

    # JPEG → numpy → BGR
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid image"}

    # YOLOv8 инференс
    results = model(img, verbose=False)[0]

    boxes = []
    classes = []
    scores = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        boxes.append([x1, y1, x2, y2])
        classes.append(cls)
        scores.append(conf)

    # Ответ включает и мета-данные кадра
    return {
        "stream_id": stream_id,
        "frame_index": index,
        "timestamp": timestamp,
        "detections": {
            "boxes": boxes,
            "classes": classes,
            "scores": scores
        }
    }
