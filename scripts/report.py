"""Prints unique visitor count and event table from the DB."""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1] if len(sys.argv) > 1 else "data/visitors.db")
print("Unique visitors:", db.execute("SELECT COUNT(*) FROM faces").fetchone()[0])
print("\nface_id | event | timestamp | image")
for r in db.execute("SELECT face_id,event_type,timestamp,image_path FROM events ORDER BY id"):
    print(" | ".join(map(str, r)))
