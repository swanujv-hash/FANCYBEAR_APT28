import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = DATA_DIR / "deeptrace.db"


def init_db():
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                officer TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'created',
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(case_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                evidence_id INTEGER NOT NULL,
                metadata_result TEXT,
                compression_result TEXT,
                landmark_result TEXT,
                audio_sync_result TEXT,
                ml_detector_result TEXT,
                priority_score REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(case_id),
                FOREIGN KEY(evidence_id) REFERENCES evidence(evidence_id)
            )
        """)

        # Lightweight migration for databases created by earlier DeepTrace versions.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis)").fetchall()}
        if "ml_detector_result" not in cols:
            conn.execute("ALTER TABLE analysis ADD COLUMN ml_detector_result TEXT")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
