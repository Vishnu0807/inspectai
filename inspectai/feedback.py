"""Store labels and notes only; never store raw uploaded images."""
import csv
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import sqlite3


SCHEMA = '''CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY, reviewed_at TEXT NOT NULL, image_hash TEXT NOT NULL,
  source TEXT NOT NULL, score REAL NOT NULL, threshold REAL NOT NULL,
  prediction TEXT NOT NULL, reviewer_label TEXT NOT NULL, note TEXT NOT NULL
)'''


def record(db_path: Path, image_hash: str, source: str, score: float,
           threshold: float, prediction: str, reviewer_label: str, note: str):
    if reviewer_label not in {'normal', 'defect'} or prediction not in {'normal', 'defect'}:
        raise ValueError('Invalid feedback label.')
    if len(note) > 300 or len(source) > 100 or len(image_hash) != 64:
        raise ValueError('Feedback fields exceed their limits.')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, timeout=10) as db:
        db.execute(SCHEMA)
        db.execute('''INSERT INTO feedback (reviewed_at,image_hash,source,score,threshold,
                    prediction,reviewer_label,note) VALUES (?,?,?,?,?,?,?,?)''',
                   (datetime.now(timezone.utc).isoformat(), image_hash, source,
                    float(score), float(threshold), prediction, reviewer_label, note.strip()))


def export_csv(db_path: Path) -> str:
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(['reviewed_at', 'image_hash', 'source', 'score', 'threshold',
                     'prediction', 'reviewer_label', 'note'])
    if db_path.exists():
        with sqlite3.connect(db_path) as db:
            db.execute(SCHEMA)
            writer.writerows(db.execute('''SELECT reviewed_at,image_hash,source,score,threshold,
                prediction,reviewer_label,note FROM feedback ORDER BY id'''))
    return out.getvalue()
