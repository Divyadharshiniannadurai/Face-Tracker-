"""Loads config.json into a simple attribute-access object."""
import json


class Config:
    DEFAULTS = {
        "source": "videos/sample.mp4", "detection_skip_frames": 2,
        "yolo_model": "models/yolov8n-face.pt", "yolo_conf": 0.5, "yolo_imgsz": 640,
        "device": "cpu", "recognition_threshold": 0.40, "min_face_size": 40,
        "min_track_hits": 3, "max_missed_detections": 6, "iou_threshold": 0.25,
        "max_id_attempts": 4, "db_path": "data/visitors.db", "log_dir": "logs",
        "display": False, "save_annotated_video": "", "rtsp_reconnect_attempts": 10,
    }

    def __init__(self, path="config.json"):
        data = dict(self.DEFAULTS)
        with open(path) as f:
            data.update(json.load(f))
        self.__dict__.update(data)
