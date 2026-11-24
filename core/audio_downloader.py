import os
import re
import shutil
import uuid
from typing import Tuple, Optional, Dict, Any

from core.config import TEMP_DIR, FFMPEG_PATH, FFMPEG_BIN_DIR, FFPROBE_PATH
from core.result_cache import build_youtube_cache_key


def is_valid_youtube_url(url: str) -> bool:
    """Memeriksa apakah URL memiliki format YouTube yang valid."""
    if not url or not isinstance(url, str):
        return False
    youtube_regex = r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$'
    return bool(re.match(youtube_regex, url.strip()))


def ensure_ffmpeg_in_runtime_path() -> Optional[str]:
    """
    Memastikan folder bin FFmpeg terdaftar di os.environ['PATH'] proses Python runtime
    sebelum yt-dlp atau subprocess lainnya dijalankan.
    """
    bin_dir = FFMPEG_BIN_DIR
    if not bin_dir and FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
        bin_dir = os.path.dirname(FFMPEG_PATH)

    if bin_dir and os.path.isdir(bin_dir):
        current_path = os.environ.get("PATH", "")
        normalized_parts = [os.path.normcase(os.path.abspath(p)) for p in current_path.split(os.pathsep) if p]
        norm_bin = os.path.normcase(os.path.abspath(bin_dir))
        if norm_bin not in normalized_parts:
            os.environ["PATH"] = bin_dir + os.pathsep + current_path
        return bin_dir
    return None


def check_ffmpeg_availability() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Memeriksa ketersediaan FFmpeg dan FFprobe di runtime aplikasi.
    Returns:
        (is_ready: bool, message: str, details: dict)
    """
    bin_dir = ensure_ffmpeg_in_runtime_path()
    f_which = shutil.which("ffmpeg")
    p_which = shutil.which("ffprobe")

    details = {
        "bin_dir": bin_dir,
        "ffmpeg_path": f_which or FFMPEG_PATH,
        "ffprobe_path": p_which or FFPROBE_PATH,
        "has_ffmpeg": bool(f_which or (FFMPEG_PATH and os.path.isfile(FFMPEG_PATH))),
        "has_ffprobe": bool(p_which or (FFPROBE_PATH and os.path.isfile(FFPROBE_PATH))),
    }

    if not details["has_ffmpeg"]:
        return False, "Executable ffmpeg tidak ditemukan. Silakan pasang FFmpeg atau daftarkan folder bin ke PATH.", details
    if not details["has_ffprobe"]:
        return False, "Executable ffprobe tidak ditemukan di folder bin. Ekstraksi metadata yt-dlp mungkin gagal.", details

    return True, "FFmpeg dan FFprobe terdeteksi dan siap digunakan.", details


def _find_cached_youtube_file(output_dir: str, file_prefix: str) -> Optional[str]:
    """Mencari file audio YouTube yang sudah ada di disk dan masih valid (> 10 KB)."""
    for ext in ('m4a', 'webm', 'opus', 'mp3', 'wav', 'ogg'):
        candidate = os.path.join(output_dir, f"{file_prefix}.{ext}")
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 10 * 1024:
            return candidate
    return None


def download_youtube_audio(
    url: str,
    output_dir: Optional[str] = None,
    convert_to_mp3: bool = False,
    use_cache: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """
    Mengunduh stream audio dari tautan video YouTube menggunakan yt-dlp.
    Secara default menghindari konversi ulang ke MP3 karena faster-whisper (PyAV) dapat membaca
    format m4a/webm/opus secara langsung sehingga lebih cepat dan hemat CPU.

    Returns:
        (success: bool, message_or_path: str, video_title: Optional[str])
    """
    if not is_valid_youtube_url(url):
        return False, "URL YouTube tidak valid. Harap periksa kembali format link.", None

    if output_dir is None:
        output_dir = TEMP_DIR
    os.makedirs(output_dir, exist_ok=True)

    file_id = build_youtube_cache_key(url) if use_cache else f"yt_{uuid.uuid4().hex[:8]}"
    if use_cache:
        existing = _find_cached_youtube_file(output_dir, file_id)
        if existing:
            return True, existing, f"YouTube (Cache Lokal: {os.path.basename(existing)})"

    out_template = os.path.join(output_dir, f"{file_id}.%(ext)s")
    target_mp3 = os.path.join(output_dir, f"{file_id}.mp3")

    try:
        import yt_dlp
    except ImportError:
        return False, "Library yt-dlp belum terinstall. Silakan install dengan 'pip install yt-dlp'.", None

    bin_dir = ensure_ffmpeg_in_runtime_path()

    ydl_opts: Dict[str, Any] = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
        'outtmpl': out_template,
        'quiet': True,
        'no_warnings': True,
    }
    if convert_to_mp3:
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128'
        }]

    if bin_dir and os.path.isdir(bin_dir):
        ydl_opts['ffmpeg_location'] = bin_dir
    elif FFMPEG_BIN_DIR and os.path.isdir(FFMPEG_BIN_DIR):
        ydl_opts['ffmpeg_location'] = FFMPEG_BIN_DIR
    elif FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
        ydl_opts['ffmpeg_location'] = os.path.dirname(FFMPEG_PATH)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'Video YouTube')

            if convert_to_mp3 and os.path.exists(target_mp3):
                return True, target_mp3, video_title

            for ext in ['m4a', 'webm', 'opus', 'mp3', 'wav', 'ogg']:
                alt_path = os.path.join(output_dir, f"{file_id}.{ext}")
                if os.path.exists(alt_path) and os.path.getsize(alt_path) > 0:
                    return True, alt_path, video_title

            return False, "File audio tidak ditemukan setelah proses pengunduhan selesai.", video_title
    except Exception as e:
        err_msg = str(e)
        err_lower = err_msg.lower()
        if "ffmpeg" in err_lower or "ffprobe" in err_lower or "avconv" in err_lower:
            return False, "FFmpeg atau FFprobe tidak dapat ditemukan oleh yt-dlp untuk konversi audio. Pastikan folder bin terpasang dengan benar.", None
        if "video unavailable" in err_lower or "private video" in err_lower:
            return False, "Video tidak tersedia, privat, atau dibatasi oleh pemilik konten.", None
        if "bot" in err_lower or "sign in" in err_lower:
            return False, "Akses YouTube dibatasi sementara oleh perlindungan bot. Silakan coba beberapa saat lagi.", None
        if "timed out" in err_lower or "urlopen error" in err_lower or "network" in err_lower or "connection" in err_lower:
            return False, "Koneksi ke YouTube terputus atau gagal dijangkau. Periksa koneksi internet Anda.", None
        return False, f"Gagal mengunduh audio YouTube: {err_msg}", None
