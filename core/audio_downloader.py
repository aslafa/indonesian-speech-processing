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


def _build_ydl_options(
    out_template: str,
    target_mp3: str,
    convert_to_mp3: bool,
    bin_dir: Optional[str],
    player_clients: Optional[list] = None,
    cookies_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Menyusun konfigurasi YoutubeDL yang tangguh dan adaptif."""
    opts: Dict[str, Any] = {
        # Prioritaskan stream audio m4a/webm, lalu fallback ke format stream campuran/audio terbaik
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
        'outtmpl': out_template,
        'quiet': True,
        'no_warnings': True,
        'cachedir': False,  # Cegah penggunaan cache signature/token kadaluarsa
        'noplaylist': True,
        'retries': 3,
        'fragment_retries': 3,
        'socket_timeout': 30,
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
            'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
        },
    }

    if player_clients:
        opts['extractor_args'] = {
            'youtube': {
                'player_client': player_clients
            }
        }

    # Dukungan file cookies jika disediakan via environment variable atau file lokal
    cookie_file = cookies_path or os.environ.get("YOUTUBE_COOKIES_FILE")
    if cookie_file and os.path.isfile(cookie_file):
        opts['cookiefile'] = cookie_file

    if convert_to_mp3:
        opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128'
        }]

    if bin_dir and os.path.isdir(bin_dir):
        opts['ffmpeg_location'] = bin_dir
    elif FFMPEG_BIN_DIR and os.path.isdir(FFMPEG_BIN_DIR):
        opts['ffmpeg_location'] = FFMPEG_BIN_DIR
    elif FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
        opts['ffmpeg_location'] = os.path.dirname(FFMPEG_PATH)

    return opts


def _extract_downloaded_filepath(output_dir: str, file_id: str, convert_to_mp3: bool, target_mp3: str) -> Optional[str]:
    """Mendeteksi file keluaran hasil unduhan yang valid di disk."""
    if convert_to_mp3 and os.path.exists(target_mp3) and os.path.getsize(target_mp3) > 1024:
        return target_mp3

    for ext in ['m4a', 'webm', 'opus', 'mp3', 'wav', 'ogg', 'mp4']:
        alt_path = os.path.join(output_dir, f"{file_id}.{ext}")
        if os.path.exists(alt_path) and os.path.getsize(alt_path) > 1024:
            return alt_path
    return None


def _format_download_error_message(err_msg: str) -> str:
    """Menerjemahkan pesan error yt-dlp ke penjelasan yang informatif dan solutif bagi pengguna."""
    err_lower = err_msg.lower()
    if "403" in err_lower or "forbidden" in err_lower:
        return (
            "YouTube menolak akses stream audio (HTTP Error 403: Forbidden). "
            "IP server hosting (Streamlit Cloud) dibatasi oleh YouTube. "
            "Solusi: Gunakan metode 'Unggah Audio' dengan mengunggah berkas MP3/M4A/WAV secara langsung."
        )
    if "ffmpeg" in err_lower or "ffprobe" in err_lower or "avconv" in err_lower:
        return (
            "FFmpeg atau FFprobe tidak dapat ditemukan oleh yt-dlp. "
            "Pastikan packages.txt di repositori menyertakan 'ffmpeg'."
        )
    if "video unavailable" in err_lower or "private video" in err_lower:
        return "Video YouTube tidak tersedia, berstatus privat, atau telah dihapus oleh pemilik konten."
    if "sign in" in err_lower or "bot" in err_lower or "confirm you're not a bot" in err_lower:
        return (
            "YouTube mendeteksi aktivitas otomatis (anti-bot) pada IP server. "
            "Silakan unduh audio di perangkat lokal Anda terlebih dahulu, lalu gunakan opsi 'Unggah Audio'."
        )
    if "timed out" in err_lower or "urlopen error" in err_lower or "network" in err_lower or "connection" in err_lower:
        return "Koneksi jaringan ke YouTube terputus atau gagal dijangkau. Periksa koneksi internet Anda."
    return f"Gagal mengunduh audio YouTube: {err_msg}"


def download_youtube_audio(
    url: str,
    output_dir: Optional[str] = None,
    convert_to_mp3: bool = False,
    use_cache: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """
    Mengunduh stream audio dari tautan video YouTube menggunakan yt-dlp dengan multi-strategy fallback.
    Secara adaptif mengatasi masalah HTTP 403 Forbidden pada datacenter/cloud (Streamlit Cloud).
    Secara default menghindari konversi ulang ke MP3 karena faster-whisper (PyAV) dapat membaca
    format m4a/webm/opus/mp4 secara langsung sehingga lebih cepat dan hemat CPU.

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

    # Strategi multi-client fallback untuk mengatasi pemblokiran HTTP 403 YouTube di cloud
    # 1. Android player client (terbukti paling resisten terhadap 403 di IP datacenter)
    # 2. Kombinasi mweb + web_embedded
    # 3. Default yt-dlp fallback
    client_strategies = [
        ['android'],
        ['mweb', 'web_embedded'],
        None,
    ]

    last_error_message = ""
    for clients in client_strategies:
        ydl_opts = _build_ydl_options(
            out_template=out_template,
            target_mp3=target_mp3,
            convert_to_mp3=convert_to_mp3,
            bin_dir=bin_dir,
            player_clients=clients,
        )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info.get('title', 'Video YouTube') if isinstance(info, dict) else 'Video YouTube'

                found_path = _extract_downloaded_filepath(output_dir, file_id, convert_to_mp3, target_mp3)
                if found_path:
                    return True, found_path, video_title

        except Exception as e:
            last_error_message = str(e)
            # Jika error adalah video unavailable/private/invalid, tidak perlu mencoba strategi client lain
            err_lower = last_error_message.lower()
            if "video unavailable" in err_lower or "private video" in err_lower or "does not exist" in err_lower:
                break
            continue

    formatted_err = _format_download_error_message(last_error_message or "Format audio tidak ditemukan.")
    return False, formatted_err, None
