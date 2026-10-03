"""ArcFace (InsightFace buffalo_l / w600k_r50) embeddings for a YOLO face box."""
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(self, device="cpu"):
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device != "cpu" else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name="buffalo_l", providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if device != "cpu" else -1, det_size=(160, 160))
        self.rec = self.app.models["recognition"]

    @staticmethod
    def crop(frame, bbox, pad=0.3):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
        y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
        return frame[y1:y2, x1:x2]

    def embed(self, frame, bbox):
        """Aligned embedding using InsightFace landmarks on the padded crop;
        falls back to a plain 112x112 resize if landmarks are not found."""
        crop = self.crop(frame, bbox)
        if crop.size == 0:
            return None
        faces = self.app.get(crop)
        if faces:
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return f.normed_embedding.astype(np.float32)
        feat = self.rec.get_feat(cv2.resize(crop, (112, 112)))[0]
        return (feat / (np.linalg.norm(feat) + 1e-9)).astype(np.float32)
