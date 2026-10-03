"""In-memory gallery of known identities, backed by the database."""
import numpy as np


class FaceGallery:
    def __init__(self, db, threshold):
        self.db, self.threshold = db, threshold
        self.embs = db.load_embeddings()  # resumes previous sessions

    def match(self, emb):
        """Returns (face_id or None, best cosine similarity)."""
        if not self.embs:
            return None, -1.0
        ids = list(self.embs.keys())
        sims = np.stack([self.embs[i] for i in ids]) @ emb
        k = int(np.argmax(sims))
        return (ids[k] if sims[k] >= self.threshold else None), float(sims[k])

    def register(self, emb, ts, image_path=""):
        fid = self.db.add_face(ts, emb, image_path)
        self.embs[fid] = emb
        return fid

    def refine(self, face_id, emb, alpha=0.1):
        """Slowly adapt stored embedding (EMA) to pose/lighting changes."""
        new = (1 - alpha) * self.embs[face_id] + alpha * emb
        new /= np.linalg.norm(new) + 1e-9
        self.embs[face_id] = new.astype(np.float32)
        self.db.update_embedding(face_id, self.embs[face_id])
