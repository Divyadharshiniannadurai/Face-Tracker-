import os
import urllib.request

URLS = [
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt",
]
DEST = "models/yolov8n-face.pt"

os.makedirs("models", exist_ok=True)
if not os.path.exists(DEST):
    for url in URLS:
        try:
            print("Downloading", url)
            urllib.request.urlretrieve(url, DEST)
            break
        except Exception as e:
            print("Failed:", e)
print("Model ready:", DEST, os.path.exists(DEST))
