"""SQLite persistence (WAL mode, commit per write => resilient to crashes)."""
import os
import sqlite3
import numpy as np


class Database:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS faces(
            face_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_seen TEXT NOT NULL,
            embedding BLOB NOT NULL,
            image_path TEXT);
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN ('entry','exit')),
            timestamp TEXT NOT NULL,
            image_path TEXT,
            track_id INTEGER,
            FOREIGN KEY(face_id) REFERENCES faces(face_id));
        """)
        self.conn.commit()

    def add_face(self, ts, emb, image_path=""):
        cur = self.conn.execute(
            "INSERT INTO faces(first_seen, embedding, image_path) VALUES(?,?,?)",
            (ts, emb.astype(np.float32).tobytes(), image_path))
        self.conn.commit()
        return cur.lastrowid

    def update_embedding(self, face_id, emb):
        self.conn.execute("UPDATE faces SET embedding=? WHERE face_id=?",
                          (emb.astype(np.float32).tobytes(), face_id))
        self.conn.commit()

    def load_embeddings(self):
        rows = self.conn.execute("SELECT face_id, embedding FROM faces").fetchall()
        return {r[0]: np.frombuffer(r[1], dtype=np.float32).copy() for r in rows}

    def add_event(self, face_id, event_type, ts, image_path, track_id):
        self.conn.execute(
            "INSERT INTO events(face_id,event_type,timestamp,image_path,track_id) VALUES(?,?,?,?,?)",
            (face_id, event_type, ts, image_path, track_id))
        self.conn.commit()

    def unique_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def close(self):
        self.conn.close()
