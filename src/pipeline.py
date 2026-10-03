"""Main processing loop: detect -> track -> recognize/register -> log entry/exit."""
import os
import time
from datetime import datetime

import cv2

from .database import Database
from .detector import FaceDetector
from .gallery import FaceGallery
from .recognizer import FaceRecognizer
from .tracker import IoUTracker


class VisitorPipeline:
    def __init__(self, cfg, logger):
        self.cfg, self.log = cfg, logger
        self.db = Database(cfg.db_path)
        self.detector = FaceDetector(cfg.yolo_model, cfg.yolo_conf, cfg.yolo_imgsz, cfg.device)
        self.recognizer = FaceRecognizer(cfg.device)
        self.gallery = FaceGallery(self.db, cfg.recognition_threshold)
        self.tracker = IoUTracker(cfg.iou_threshold, cfg.max_missed_detections)
        self.active = {}          # face_id -> set(track_id) currently in frame
        self.stop = False
        self.last_t = 0.0
        self.log.info("SYSTEM_START | source=%s | skip=%d | known_faces=%d",
                      cfg.source, cfg.detection_skip_frames, len(self.gallery.embs))

    # ---------- helpers ----------
    @staticmethod
    def _ts():
        return datetime.now().isoformat(timespec="milliseconds")

    def _save_crop(self, kind, face_id, track, ts):
        day = ts[:10]
        folder = os.path.join(self.cfg.log_dir, kind, day)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{ts[11:23].replace(':', '-')}_face{face_id}_{"entry" if kind == "entries" else "exit"}.jpg")
        if track.last_crop is not None and track.last_crop.size:
            cv2.imwrite(path, track.last_crop)
        return path

    def _event(self, kind, track, video_t):
        """kind: 'entries' | 'exits'. Exactly one DB row + one image + one log line."""
        ts = self._ts()
        path = self._save_crop(kind, track.face_id, track, ts)
        etype = "entry" if kind == "entries" else "exit"
        self.db.add_event(track.face_id, etype, ts, path, track.track_id)
        self.log.info("%s | face_id=%d | track=%d | video_t=%.2fs | img=%s",
                      etype.upper(), track.face_id, track.track_id, video_t, path)

    # ---------- identity ----------
    def _identify(self, frame, track, video_t):
        emb = self.recognizer.embed(frame, track.bbox)
        if emb is None:
            return
        self.log.info("EMBEDDING_GENERATED | track=%d | dim=%d", track.track_id, emb.shape[0])
        fid, sim = self.gallery.match(emb)
        track.id_attempts += 1
        if fid is not None:
            track.face_id = fid
            self.gallery.refine(fid, emb)
            self.log.info("RECOGNIZED | face_id=%d | track=%d | sim=%.3f", fid, track.track_id, sim)
        elif sim >= self.cfg.recognition_threshold - 0.1 and \
                track.id_attempts < self.cfg.max_id_attempts:
            self.log.info("AMBIGUOUS_MATCH | track=%d | sim=%.3f | retrying", track.track_id, sim)
            return  # borderline: wait for a better frame before registering
        else:
            ts = self._ts()
            fid = self.gallery.register(emb, ts)
            track.face_id = fid
            self.log.info("REGISTERED | face_id=%d | track=%d | unique_count=%d",
                          fid, track.track_id, self.db.unique_count())
        # Face already visible via another (fragmented) track -> same visit, no new entry.
        if self.active.get(track.face_id):
            track.entered = True
            self.active[track.face_id].add(track.track_id)
            self.log.info("TRACK_MERGED | face_id=%d | track=%d", track.face_id, track.track_id)
        else:
            self._event("entries", track, video_t)
            self.active[track.face_id] = {track.track_id}
            track.entered = True

    def _handle_lost(self, track, video_t):
        if track.entered and track.face_id is not None:
            ids = self.active.get(track.face_id, set())
            ids.discard(track.track_id)
            if not ids:
                self.active.pop(track.face_id, None)
                self._event("exits", track, video_t)
        else:
            self.log.info("TRACK_DROPPED_UNIDENTIFIED | track=%d", track.track_id)

    # ---------- main loop ----------
    def _open(self):
        src = self.cfg.source
        cap = cv2.VideoCapture(src)
        return cap, str(src).lower().startswith(("rtsp://", "http://", "rtmp://"))

    def run(self):
        cap, is_stream = self._open()
        if not cap.isOpened():
            self.log.error("Cannot open source %s", self.cfg.source)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer, frame_idx, last_det, retries = None, 0, 0, 0
        start = time.time()
        try:
            while not self.stop:
                ok, frame = cap.read()
                if not ok:
                    if is_stream and retries < self.cfg.rtsp_reconnect_attempts:
                        retries += 1
                        self.log.warning("STREAM_RECONNECT | attempt=%d", retries)
                        cap.release(); time.sleep(2)
                        cap, _ = self._open()
                        continue
                    break
                retries = 0
                video_t = frame_idx / fps
                self.last_t = video_t
                if frame_idx % (self.cfg.detection_skip_frames + 1) == 0:
                    self._detection_cycle(frame, frame_idx - last_det, video_t)
                    last_det = frame_idx
                else:
                    self.tracker.predict()
                annotated = self._annotate(frame)
                if self.cfg.save_annotated_video:
                    if writer is None:
                        os.makedirs(os.path.dirname(self.cfg.save_annotated_video) or ".", exist_ok=True)
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(self.cfg.save_annotated_video,
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(annotated)
                if self.cfg.display:
                    cv2.imshow("Visitor Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_idx += 1
        finally:
            self._shutdown(cap, writer, frame_idx, time.time() - start)

    def _detection_cycle(self, frame, gap, video_t):
        dets = self.detector.detect(frame)
        lost = self.tracker.update(dets, max(1, gap))
        for t in self.tracker.tracks:
            if not t.updated:
                continue
            crop = FaceRecognizer.crop(frame, t.bbox, pad=0.15)
            area = (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
            if crop.size and area >= 0.8 * t.best_area:  # keep a recent, large crop
                t.last_crop, t.best_area = crop.copy(), max(area, t.best_area)
            if t.face_id is None and t.hits >= self.cfg.min_track_hits \
                    and t.size >= self.cfg.min_face_size:
                self._identify(frame, t, video_t)
        for t in lost:
            self._handle_lost(t, video_t)

    def _annotate(self, frame):
        out = frame.copy()
        for t in self.tracker.tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            color = (0, 200, 0) if t.face_id else (0, 165, 255)
            label = f"ID {t.face_id}" if t.face_id else f"trk {t.track_id}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(out, f"Unique visitors: {self.db.unique_count()}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out

    def _shutdown(self, cap, writer, frames, elapsed):
        # Flush: every face still in frame gets its exit event.
        for t in list(self.tracker.tracks):
            self._handle_lost(t, self.last_t)
        self.tracker.tracks.clear()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        self.log.info("SYSTEM_STOP | frames=%d | elapsed=%.1fs | UNIQUE_VISITORS=%d",
                      frames, elapsed, self.db.unique_count())
        self.db.close()
