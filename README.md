uv init
uv venv
uv sync
uvicorn main:app --reload --port 8001
curl -X POST "http://localhost:8001/detect" -F "file=@image.jpg" --output result.jpg
curl -X POST "http://localhost:8001/detect_video" -F "file=@video.mov" --output result.mov