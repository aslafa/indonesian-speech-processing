import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from core.config import CACHE_DIR


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """
    Menghitung SHA-256 file audio secara bertahap (chunked) agar hemat RAM pada file besar.
    """
    if not file_path or not os.path.isfile(file_path):
        return None
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                buf = f.read(chunk_size)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except Exception:
        return None


def build_pipeline_cache_key(
    audio_hash: str,
    whisper_model: str,
    enable_cleaning: bool,
    summary_mode: str,
) -> str:
    """
    Membuat kunci cache deterministik berdasarkan hash audio dan parameter pemrosesan.
    """
    raw = f"{audio_hash}|whisper={whisper_model}|clean={enable_cleaning}|mode={summary_mode}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_transcript_cache_key(audio_hash: str, whisper_model: str) -> str:
    """Membuat kunci cache khusus hasil transkripsi Whisper."""
    raw = f"tr|{audio_hash}|whisper={whisper_model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_youtube_cache_key(url: str) -> str:
    """Membuat kunci cache berdasarkan URL YouTube yang dinormalisasi."""
    norm = url.strip()
    return "yt_" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]


def _cache_file_path(cache_key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{cache_key}.json")


def get_cached_entry(cache_key: str) -> Optional[Dict[str, Any]]:
    """Mengambil hasil cache JSON jika tersedia dan valid."""
    path = _cache_file_path(cache_key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("raw_transcript") is not None:
            return data
        return None
    except Exception:
        return None


def save_cached_entry(cache_key: str, payload: Dict[str, Any]) -> Optional[str]:
    """Menyimpan hasil pemrosesan riil ke disk cache."""
    path = _cache_file_path(cache_key)
    try:
        data = dict(payload)
        data["cached_at"] = datetime.now().isoformat()
        data["cache_key"] = cache_key
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None

