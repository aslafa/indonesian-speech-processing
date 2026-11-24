import gc
import os
import time
import threading
from typing import Dict, Any

from core.config import TEMP_DIR, RAM_HIGH_WATERMARK_PCT, RAM_MIN_AVAILABLE_MB

# Lock global untuk mencegah dua proses inference berjalan bersamaan (menghindari lonjakan RAM & contention CPU)
INFERENCE_LOCK = threading.Lock()


def get_memory_stats() -> Dict[str, Any]:
    """
    Mengambil statistik penggunaan RAM proses saat ini dan RAM sistem secara keseluruhan.
    """
    stats: Dict[str, Any] = {
        "process_rss_mb": 0.0,
        "system_total_gb": 0.0,
        "system_available_mb": 0.0,
        "system_percent": 0.0,
        "is_critical": False,
    }
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        vm = psutil.virtual_memory()
        stats["process_rss_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)
        stats["system_total_gb"] = round(vm.total / (1024 ** 3), 2)
        stats["system_available_mb"] = round(vm.available / (1024 ** 2), 1)
        stats["system_percent"] = round(vm.percent, 1)
        stats["is_critical"] = (
            vm.percent >= RAM_HIGH_WATERMARK_PCT
            or stats["system_available_mb"] < RAM_MIN_AVAILABLE_MB
        )
    except Exception:
        pass
    return stats


def force_gc() -> None:
    """Memicu garbage collection untuk melepaskan objek audio dan tensor sementara."""
    gc.collect()


def cleanup_stale_temp_files(max_age_sec: int = 3600, exclude_names: tuple = ("test_tone.wav",)) -> int:
    """
    Membersihkan file temporary lama di TEMP_DIR tanpa menghapus file yang masih dibutuhkan.
    """
    if not os.path.isdir(TEMP_DIR):
        return 0
    now = time.time()
    removed = 0
    try:
        for name in os.listdir(TEMP_DIR):
            if name in exclude_names:
                continue
            full_path = os.path.join(TEMP_DIR, name)
            if os.path.isfile(full_path):
                # Jangan hapus file audio YouTube berukuran besar yang baru saja dibuat (< max_age_sec)
                try:
                    age = now - os.path.getmtime(full_path)
                    if age > max_age_sec and name.startswith(("upload_", "yt_", "tmp")):
                        os.remove(full_path)
                        removed += 1
                except Exception:
                    pass
    except Exception:
        pass
    return removed

