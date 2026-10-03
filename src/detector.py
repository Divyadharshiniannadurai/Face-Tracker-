"""YOLOv8 face detector."""
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path, conf=0.5, imgsz=640, device="cpu"):
        self.model = YOLO(model_path)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, frame):
        """Returns list of (x1, y1, x2, y2, conf)."""
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out
