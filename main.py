from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import tempfile
import os

app = FastAPI()

# Загружаем модель при старте
model = YOLO('yolov8n.pt')

def draw_boxes(img, results):
    """Рисует bounding boxes на изображении"""
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Читаем изображение
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Детекция объектов
    results = model(img)[0]
    
    # Рисуем bounding boxes
    img = draw_boxes(img, results)
    
    # Конвертируем обратно в bytes
    _, img_encoded = cv2.imencode('.jpg', img)
    
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    # Определяем расширение файла
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm']:
        return {"error": "Unsupported video format"}
    
    # Сохраняем загруженное видео во временный файл с оригинальным расширением
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
        contents = await file.read()
        temp_input.write(contents)
        temp_input_path = temp_input.name
    
    # Создаем временный файл для выходного видео (mp4)
    temp_output_path = tempfile.mktemp(suffix='.mp4')
    
    try:
        # Открываем видео
        cap = cv2.VideoCapture(temp_input_path)
        
        if not cap.isOpened():
            return {"error": "Failed to open video file"}
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создаем writer для выходного видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        # Обрабатываем каждый кадр
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекция объектов
            results = model(frame)[0]
            
            # Рисуем bounding boxes
            frame = draw_boxes(frame, results)
            
            # Записываем кадр
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Читаем обработанное видео
        with open(temp_output_path, 'rb') as f:
            video_bytes = f.read()
        
        return StreamingResponse(BytesIO(video_bytes), media_type="video/mp4")
    
    finally:
        # Удаляем временные файлы
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

@app.get("/")
def root():
    return {
        "message": "YOLO Detection API",
        "endpoints": {
            "POST /detect": "Upload image for object detection",
            "POST /detect_video": "Upload video for object detection"
        }
    }